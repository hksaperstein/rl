# toy_env: a CPU-only, physics-free proxy environment for fast RL prototyping

## What this is

`toy_env/` (repo root, sibling to `tasks/`, `vision/`, `perception/`) is a new,
deliberately separate CPU-only Python package: a pure-kinematics (no contact,
no collision, no friction, no mass/inertia/torque) 7-joint arm reaching
toward and "grasping" a target, exposed as a standard Gymnasium
(`reset()`/`step()`/`observation_space`/`action_space`) environment. It exists
to answer this project's own repeatedly-stated bottleneck all session:
**GPU cost/turnaround** — a shared single-GPU cloud quota, hours-long Isaac
Lab training runs, real dollar cost per experiment (see e.g. this session's
own d8-antipodal cost accounting in
[[d8-antipodal-grasp-quality]]: ~$3.2 of cloud spend and multiple hours of
wall-clock time for one experiment's two conditions). `toy_env/` lets
algorithm/exploration/action-space questions be iterated in **seconds to
low-single-digit minutes on a Raspberry Pi**, with zero cloud cost, before
committing real Isaac Sim GPU time to anything that looks promising.

**This is infrastructure/tooling, not an RL experiment** — it does not go
through this project's Tier 1 spec/plan gate (`CLAUDE.md`'s Workflow
section), but per the same document's continuous-documentation convention,
it's recorded here rather than left undocumented.

## What this is explicitly NOT

Read this section before trusting any result produced using `toy_env/` for a
real decision:

- **No contact physics, no friction, no real grasp mechanics.** A "grasp" in
  `toy_env` is a kinematic proxy: the object rigidly attaches to the
  end-effector tip position the instant `distance < GRIP_DIST_THRESHOLD` AND
  the gripper action is "closed," with no contact force, no antipodal/
  force-closure condition, no possibility of a partial or misaligned grasp.
  Compare this to what a *real* grasp actually requires in this project's own
  findings —
  [[grasp-mechanics-antipodal-vs-magnitude]] and the antipodal/force-closure
  mechanism in [[d8-antipodal-grasp-quality]] — and it's clear this proxy
  cannot say anything about grasp-quality mechanisms, only about reaching/
  action-space dynamics upstream of contact.
- **No dynamics.** Forward kinematics only (`toy_env/kinematic_arm.py`) — no
  mass, no inertia, no torque limits, no joint-velocity-dependent effects.
- **A result here is a hypothesis generator, not a conclusion.** Anything
  interesting found in `toy_env` — an action-space effect, an algorithm
  comparison, an exploration-bonus idea — still needs re-verification in the
  real Isaac Lab simulator before being trusted for this project's actual
  Franka/dice work. Do not treat a toy_env finding as itself dispositive.

## Design: reproducing the real action-space finding, not just "an arm"

The whole point of this environment (per its own design brief) is to
preserve the *specific* property this project's own real Franka/d8
investigation diagnosed as the root cause of a training failure — not merely
to simulate "an arm reaching a target." See
[[d8-antipodal-grasp-quality]]'s "Root cause investigation" section (Finding
3) for the real finding: under **joint-space/absolute-target control**, a
policy transiently discovers how to approach the object (peaking early in
training) and then **regresses away from that behavior** over the remaining
training, converging to a "hover-near-but-never-touch" local optimum;
**task-space/differential-IK control** does not show this regression — the
`reaching_object` reward rises and *stays* high. The literature grounding for
this (cited in that article, existence/accuracy-checked per this project's
own citation practice) is Martín-Martín et al. IROS 2019 (arXiv:1906.08880),
Varin et al. IROS 2019 (arXiv:1908.08659), and Hsu et al. 2020
(arXiv:2009.10897, a PPO-specific credit-assignment/entropy-narrowing failure
mode).

`toy_env/kinematic_arm.py` implements a 7-joint revolute chain (matching
Franka Panda's own joint count, without attempting to match its exact DH
parameters — that precision isn't needed for what this environment is used
for) via forward kinematics (`forward_kinematics`) and a purely numerical
(finite-difference) position Jacobian (`jacobian_position`), used for a basic
Moore-Penrose pseudo-inverse IK solve.

`toy_env/arm_reach_env.py`'s `ArmReachEnv` implements **three action modes**,
chosen specifically to isolate configuration-dependent vs.
configuration-independent action semantics:

1. **`"absolute"`** — each action component maps to a fixed *absolute target
   joint angle*; the arm moves toward that target at a bounded max velocity
   each step. The toy-scale analogue of Isaac Lab's `JointPositionActionCfg`
   (offset-from-default-pose target semantics): the same action value always
   means the same target, but the resulting *motion from wherever the arm
   currently is* is heavily configuration-dependent.
2. **`"relative"`** — each action component maps to a fixed-scale *joint-angle
   delta* added directly to the current joint angles. The toy-scale analogue
   of Isaac Lab's `RelativeJointPositionActionCfg` — configuration-independent
   in joint-space, though still configuration-dependent in Cartesian/
   end-effector space (same joint delta, different EE motion at different
   poses, via the pose-dependent Jacobian) — a distinction deliberately
   preserved between this mode and task-space below.
3. **`"task_space"`** — action components are a desired end-effector Cartesian
   velocity, converted to a joint-space delta via `pinv(jacobian_position(theta))
   @ v_desired`. The toy-scale analogue of Isaac Lab's
   `DifferentialInverseKinematicsActionCfg`: the action's effect on the end
   effector is ~configuration-independent by construction. Position-only (no
   orientation control) — a deliberate simplification, matching the design
   brief's "even a basic Jacobian pseudo-inverse solve is fine."

All three modes share the same staged reach→grip→lift reward (dense reach
distance shaping, a flat grip bonus requiring both proximity and a
closed-gripper action, a lift bonus requiring grip plus height gain — built
additively so a later stage is never worth less than fully achieving an
earlier one, following this project's own non-decreasing-staged-reward
precedent from AR4-era Experiment 25 and
[[staged-reward-co-satisfiability]]) and the same observation/episode
structure — only the action interpretation differs, so a training-curve
comparison across modes isolates the action-space effect specifically.

## Gymnasium interface, and why that specific choice

`ArmReachEnv` implements the standard `gymnasium.Env` API
(`reset()`/`step()`/`observation_space`/`action_space`), verified via
`gymnasium.utils.env_checker.check_env` for all three action modes. This
makes it a drop-in fit for Stable-Baselines3's ready-made PPO and SAC
implementations with zero additional integration work — directly supporting
the PPO-vs-SAC question raised this session, without needing any Isaac-Lab-
specific plumbing. `toy_env/train_demo.py --algo sac` is already wired
(untested as of this writing — the worked demonstration below only exercises
PPO, per the design brief's own "PPO, your call" framing; SAC is available
for a follow-up run without further code changes).

## Visualization

`toy_env/visualize.py` — pure `matplotlib` (`mpl_toolkits.mplot3d`), no Isaac
Sim/rendering dependency:

- `plot_arm_pose_3d(env, ...)`: a static 3D plot of the current arm pose
  (base/joints/end-effector distinguished by marker), the target, the
  (kinematically-attached-or-not) object, the episode's trajectory trace so
  far, and optional faint "ghost" poses from earlier in the episode.
- `render_episode_gif(env, policy=None, ...)`: runs one full episode (random
  actions, or a supplied policy — including a Stable-Baselines3 model's
  `.predict`) and saves an animated GIF via `matplotlib.animation.PillowWriter`
  (no `ffmpeg` dependency), so a rollout can actually be watched — this
  project's own "watch it work, don't trust a scalar" verification culture,
  at zero GPU cost.

## Environment/runtime: a third, deliberately separate Python environment

This repo already has two Python runtimes with a strict "path decides the
interpreter" convention (`vision/.venv` for the vision subtree's PyTorch
cu128 GPU build; Isaac Lab's own `isaaclab.sh -p` for everything Isaac-Sim-
touching). `toy_env/` adds a **third**, `toy_env/.venv`, deliberately
isolated from both — it needs Gymnasium/Stable-Baselines3/matplotlib/a CPU
build of PyTorch, none of which belong in either existing environment, and it
must never accidentally launch anything Isaac-Sim-related or contend for the
desktop/cloud GPU this project's other work depends on. `toy_env/` code
should never be run via `isaaclab.sh` or `vision/.venv`, and neither of those
should ever import from `toy_env/`.

One real operational gotcha hit while setting this up, worth recording:
**plain `pip install torch` on this project's Raspberry Pi host pulls in
PyTorch's default aarch64 wheel, which — unlike x86 — bundles NVIDIA
`cuda-toolkit`/`cudnn` dependencies (~1.5GB) even though the Pi has no NVIDIA
GPU at all.** Installing from PyTorch's own CPU-only index
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) avoids
this entirely (a ~155MB CPU-only wheel instead) — `toy_env/requirements.txt`
documents this explicitly so it isn't rediscovered the hard way next time.
Relatedly, this Pi's `/tmp` is a small (~900MB) `tmpfs`, not real disk — a
large `pip install` (e.g. downloading `torch`'s ~427MB CUDA-bundled wheel
before the CPU-only fix was applied) can fail with `No space left on device`
even though the root filesystem has plenty of headroom; set `TMPDIR` to a
directory on the real disk (this repo used `/home/pi/pip_tmp`) if that
happens again.

## Worked demonstration: does the toy env reproduce the real finding?

`toy_env/train_demo.py` trains a real Stable-Baselines3 PPO agent under each
action mode on this environment, with a separate deterministic fixed-seed
evaluation loop (`EvalRecorderCallback`) recording a training-time curve of
`mean_min_dist` (closest approach to target per eval episode — this toy
environment's own analogue of the real experiment's `reaching_object` reward
curve) at regular checkpoints throughout training, specifically so a
rise-then-decay shape (the real pathology) or its absence can be read off
directly, not just inferred from a single final number.

**Real run, 1 seed each, 100,000 timesteps, PPO, `n_envs=4`
(`toy_env/runs/comparison_ppo_seed0.json`):**

| action mode | best `mean_min_dist` | best reached @ | final `mean_min_dist` | regression (final − best) | final success_rate |
|---|---|---|---|---|---|
| `absolute`   | 0.0386 | t=80,016 (of 100,000) | 0.0536 | +0.0150 (**+38.7%** of best) | 0.000 |
| `relative`   | 0.0532 | t=46,676               | 0.0689 | +0.0157 (**+29.5%** of best) | 0.000 |
| `task_space` | 0.0354 | t=73,348               | 0.0592 | +0.0238 (**+67.1%** of best) | 0.000 |

(`mean_min_dist` = mean, across 5 fixed-seed deterministic eval episodes, of
the closest end-effector-to-target approach that episode — see
`toy_env/train_demo.py`'s `EvalRecorderCallback`. Plot:
`toy_env/renders/comparison_seed0.png`, generated but not committed —
gitignored alongside the rest of `toy_env/renders/`.)

**Honest verdict: this run does NOT reproduce the real Isaac Lab pathology,
and the toy environment's own limits are the most likely reason why —
report this as the real, if negative, finding it is, not a success.**

- All three modes show the *same qualitative shape*: a fast initial descent
  (random-policy baseline ~0.90-0.95m down to sub-0.2m within the first
  ~20,000 steps), then a noisy plateau in the 0.04-0.10m range with no clean
  monotonic convergence — ordinary late-training PPO variance, not a
  meaningful late-training collapse. None of the three regressions (29-67%
  of each mode's own best) come close to the real experiment's actual
  magnitude — there, `reaching_object` peaked at 0.60 and finished at
  0.0957, an ~84% collapse essentially back to the random-policy floor
  ([[d8-antipodal-grasp-quality]]'s Finding 3 table). All three toy-mode
  regressions here are far smaller and look like normal training noise
  around a converged plateau, not a distinct failure mode.
- **The ordering is the opposite of the hypothesis, not just "absent."** If
  anything, `task_space` shows the *largest* relative regression (67.1%) of
  the three, and `absolute`/`relative` are close to each other (38.7% vs.
  29.5%) — task-space was expected to be the most stable of the three, not
  the least. Given this is a single seed per mode with a clearly noisy
  curve (see the plot — every mode's curve bounces up and down by 0.02-0.05m
  step to step throughout the plateau), this ordering should be read as
  noise, not as evidence *against* the real finding — but it is explicitly
  not a directional confirmation either, and this article does not round it
  up into one.
- **`success_rate` is 0.000 for every mode at every checkpoint** — none of
  the three ever completed a sustained grip+lift within 100,000 timesteps,
  despite all three reliably reaching within a few centimeters of
  `GRIP_DIST_THRESHOLD=0.05m`. This is itself a real, separate finding worth
  naming: getting close is not the same as reliably coordinating "be within
  5cm" with "gripper action closed" at the same instant — a small but real
  echo of this project's own much larger, much more thoroughly-characterized
  [[reach-grasp-lift-gap]], though this toy run does not attempt to
  root-cause it (no video/contact-level diagnostic was run here, and the
  toy grasp mechanism is a kinematic proxy, not a real contact check — see
  the scope-limits section above).
- **The most likely explanation for the non-reproduction is scale, not that
  the real finding is wrong.** The real experiment's own root-cause analysis
  (Finding 3, [[d8-antipodal-grasp-quality]]) explicitly attributes the
  joint-space regression to a PPO entropy-narrowing dynamic (Hsu et al. 2020)
  that plays out over *1500 on-policy PPO iterations across thousands of
  parallel environments* — many millions of environment steps total. This
  demo ran 100,000 total environment steps per mode — roughly two to three
  orders of magnitude less data — with a much smaller/simpler 7-DOF
  kinematic chain, a much smaller policy network (SB3 `MlpPolicy` defaults),
  and only `n_envs=4` (a CPU/wall-clock constraint of this task's own host,
  a 4-core Raspberry Pi with ~1GB free RAM — see the runtime section above).
  It is plausible the same qualitative regression would emerge in this same
  toy environment given a much longer run (more total timesteps, more
  parallel envs) — that is a concrete, well-motivated next step if this
  environment is revisited, not something this demo's own 1-seed/100k-step
  run can rule in or out.
- **This is a hypothesis-generator finding about the tool itself, exactly as
  this article's own scope section warns**: at this run's scale, `toy_env`
  is validated as capable of training real policies end-to-end across all
  three action modes cheaply (roughly 7-22 minutes per 100k-timestep
  condition on a Raspberry Pi, no GPU, no cloud cost — `task_space` is the
  slowest of the three, ~3x `absolute`/`relative`'s wall time, from its own
  per-step Jacobian pseudo-inverse solve), and of measuring a real,
  interpretable training-time curve — but this specific run does not, on its
  own, corroborate or refute the real Isaac Lab action-space finding. Scaling
  up the run (more timesteps, more parallel envs, multiple seeds) is the
  natural next test if this question is revisited, and is not decided or
  started here.

## Scaled-up validation run (2026-07-21): 500k timesteps/mode, 5x scale — settles the open question, in the negative

The 100k-step demo above left an explicit open question: was the
non-reproduction "most plausibly explained by insufficient training scale"?
This run answers that directly by re-running the identical comparison at
**500,000 timesteps per mode (5x the original), single seed (seed=0), same
`n_envs=4`, same PPO hyperparameters** — pure CPU wall-clock cost, no GPU/cloud
involved, dispatched sequentially (`absolute` → `relative` → `task_space`, not
in parallel) specifically to avoid oversubscribing this Pi's 4 cores and
~1.8GB RAM; live `free -h`/`ps` monitoring throughout confirmed memory stayed
bounded (~1.1-1.2GB used, no swim into swap, no OOM) the entire ~2h39m run.

**Results (`toy_env/runs_500k/comparison_ppo_seed0.json`, local-only per this
environment's own gitignored-runs convention — see below):**

| action mode | best `mean_min_dist` | best reached @ | final `mean_min_dist` | regression (final − best) | final success_rate | wall-clock |
|---|---|---|---|---|---|---|
| `absolute`   | 0.0401 | t=133,344 (of 500,000) | 0.0430 | +0.0029 (**+7.1%** of best) | 0.200 | 1852s (~31 min) |
| `relative`   | 0.0370 | t=300,024               | 0.0470 | +0.0100 (**+27.0%** of best) | 0.200 | 1862s (~31 min) |
| `task_space` | 0.0383 | t=466,704 (near the very end) | 0.0425 | +0.0042 (**+11.1%** of best) | 0.000 | 5803s (~97 min) |

(Same `mean_min_dist`/`success_rate` metrics as the 100k run above, same
`EvalRecorderCallback`, 5 fixed-seed eval episodes per checkpoint, ~15
checkpoints across training. `task_space` remains ~3.1x slower per step than
the other two, consistent with the original run's own observed ~3x ratio.)

**Two clear, load-bearing findings, neither of which is what the real Isaac
Lab experiment predicts:**

1. **No collapse pattern at 5x scale either — if anything, regressions got
   *smaller*, not larger, as scale increased.** All three modes' regressions
   here (7.1% / 27.0% / 11.1% of best) are comparable to or smaller than the
   same modes' own 100k-scale regressions (38.7% / 29.5% / 67.1%). The real
   Isaac Lab experiment's own signature is an ~84% collapse that gets *worse*
   with more training, not better — this run shows the opposite trend
   direction with 5x more data, which argues against "just needs more scale"
   as the explanation for non-reproduction, at least within this proxy's own
   dynamics.
2. **A genuinely new, real result: `success_rate` (full grip+lift) is nonzero
   for the first time in this environment's history — but only in the
   joint-space modes, never in `task_space`.** `task_space` is a clean **0/16
   checkpoints** (55 eval episodes across the entire back half of training,
   t=166,680 through t=500,040) despite having comparable-to-best `reach`
   performance (its own best `mean_min_dist`, 0.0383, is the best of all three
   modes at this scale). `absolute` succeeds at 2/16 checkpoints (rate 0.2
   each); `relative` succeeds at 5/16 checkpoints (rates 0.2-0.4, its most
   frequent success mode overall). **This is the opposite of what the real
   finding would predict** — task-space/differential-IK control is the mode
   expected to be the more reliable one for actually completing manipulation,
   not the one that never once completes it while the two joint-space modes
   both do.

**Honest verdict: this settles the open question, and settles it in the
negative.** `toy_env`, even at a meaningful (5x) scale-up run entirely on this
Pi's CPU, does **not** reproduce the real Isaac Lab action-space pathology
(transient reach-discovery-then-abandonment under absolute joint control,
stability under task-space control) — and on the one metric where the three
modes did meaningfully diverge at this scale (success rate), the direction is
inverted relative to the real finding, not merely absent or noisy.

**Most plausible root cause, per this run's own evidence, not just this
proxy's documented scope limits in the abstract:**

- **No real contact/grasp mechanics is very likely the actual gate, not
  insufficient steps.** The real Isaac Lab regression this environment tries
  to reproduce is root-caused (see
  [[d8-antipodal-grasp-quality]]'s "Root cause investigation" section, Finding
  1) to the policy converging to **never make contact with the object at
  all** under joint-space control — a genuine contact-frequency effect,
  measured directly from real contact-sensor data, that only exists because
  the real environment has real contact physics to avoid or seek. This proxy's
  "grasp" is a trivial `dist < threshold AND gripper closed` check with no
  contact force and no possibility of a near-miss — there is no equivalent
  "avoid contact entirely" local optimum available to a policy here, since
  there is no contact to avoid. If this diagnosis is right, no amount of
  additional scale in this specific proxy would ever surface the real
  pathology, because the mechanism it depends on isn't represented.
- **`task_space`'s own zero-success result is plausibly an artifact of this
  proxy's crude differential-IK implementation, not a real task-space
  limitation.** This toy `task_space` mode has no orientation control and
  uses a numerical (finite-difference) pseudo-inverse Jacobian with a
  per-step clip — a much cruder controller than Isaac Lab's own tuned
  `DifferentialInverseKinematicsActionCfg`. Holding the end effector inside a
  narrow 5cm capture radius for 10 consecutive steps via coarse per-step
  Cartesian-velocity commands may simply be a harder control problem for this
  specific crude IK solve than reaching the same radius via direct joint
  targets — a fact about this proxy's own grip mechanic and IK quality, not
  evidence about real task-space control.
- **500k steps/mode is still ~2-3 orders of magnitude below the real
  experiment's own total budget** (1500 PPO iterations × thousands of
  parallel envs ≈ 10⁸ env-steps) — a further scale-up is not ruled out in
  principle, but the fact that this run's own regressions shrank rather than
  grew with 5x more data is itself evidence against "more scale would reveal
  the same collapse" in this specific proxy, not evidence for it.

**Bottom line on the tool's usefulness — real and valuable, but narrower than
originally hoped:** `toy_env` is validated as a genuinely working, fast,
real-PPO CPU training harness (trains all three action modes end-to-end,
produces interpretable learning curves, and at this larger scale even
produces real task successes for the first time) — useful as a general
algorithm/exploration prototyping tool. It is **not** validated as a cheap
proxy for the specific action-space-dependent collapse finding in
[[d8-antipodal-grasp-quality]], and this run's own evidence (regressions
shrinking with scale, plus the inverted success-rate pattern) makes "just run
it longer" an unpromising next step for that specific question. If this axis
is revisited, the honest options are: (a) treat the real collapse as
something that can only be tested in the real Isaac Lab simulator, since it
appears to depend on genuine contact dynamics this proxy cannot represent; or
(b) rebuild this proxy with a much more faithful grasp mechanic (something
closer to a real antipodal/force-closure contact check) and/or a properly
vectorized (multi-core or GPU-batched) environment and a much larger step
budget — neither of which is decided or started here, per this task's own
scope (measure and report, not redesign).

**Where the artifacts live:** `toy_env/runs_500k/` (models + per-mode/
combined history JSON), gitignored alongside `toy_env/runs/` per this
environment's existing "run artifacts are local-only, results get written up
here" convention — not committed to git, same as the original 100k run's own
`toy_env/runs/` outputs.

## Where this lives, and why not under `tasks/`

`toy_env/` is a new top-level directory, a sibling to `tasks/`/`vision/`/
`perception/`, not nested under `tasks/franka/`. This is a genuinely
different kind of code from the rest of this repo (no Isaac Sim/Lab
dependency at all, its own venv, its own much lighter-weight verification
loop) — mixing it into `tasks/franka/` would make it look like part of the
Franka task family it's meant to help prototype *for*, when it's actually a
general-purpose, arm-and-task-agnostic tool that happens to be motivated by
(and validated against) the Franka action-space finding.

## Related

[[action-space-design]] (the joint-space-vs-task-space axis this environment
exists to let be tested cheaply), [[d8-antipodal-grasp-quality]] (the real
Isaac Lab finding — root-cause section — this environment tries to reproduce
at zero GPU cost), [[staged-reward-co-satisfiability]] (the non-decreasing
staged-reward precedent this environment's reach→grip→lift reward follows),
[[grasp-mechanics-antipodal-vs-magnitude]] (why this environment's own
"grasp" is a proxy, not real grasp mechanics), [[cloud-training]] (the cost/
turnaround problem this environment exists to work around for early-stage
prototyping specifically).
