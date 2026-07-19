# tasks/franka/distillation.py
"""Pure-torch (+ rsl_rl) mechanics for distilling frozen per-shape Franka
die-lift specialists into one unified policy.

Task 4 of `docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
distillation.md` ("Distillation pipeline (local, no GPU training yet)").
Design rationale: `docs/superpowers/specs/2026-07-16-unified-multi-die-
specialist-distillation-design.md` (UniDexGrasp++'s GiGSL specialist->
distill->iterate pattern, Wan et al., ICCV 2023, arXiv:2304.00464).

NO isaaclab/pxr/AppLauncher imports anywhere in this module - follows the
same importable-without-Isaac-Sim split `tasks/franka/shape_observations.py`
already established for observation math and `tasks/franka/lift_reward.py`
for reward math, extended here to the distillation pipeline's own
rollout-collection/relabeling/BC-loss mechanics, so this task's own
required unit tests (`tests/test_distillation_data_collection.py`) can
exercise this logic directly against stub policies/envs, no Isaac Sim
launch needed. `scripts/distill_specialists.py` is the thin CLI entry
point that wires this module to argparse, `isaaclab.app.AppLauncher`, and
(only for a real, non-dry-run run) the real Isaac Lab single-shape envs.

`rsl_rl` (`rsl_rl.modules.ActorCritic`, the `rsl-rl-lib` PyPI package) is
NOT an isaaclab/pxr/Kit-runtime-coupled import - confirmed directly
(2026-07-19) that `from rsl_rl.modules import ActorCritic` succeeds via
`/home/saps/IsaacLab/_isaac_sim/python.sh` WITHOUT first constructing an
`isaaclab.app.AppLauncher`, the same no-Isaac-Sim-launch-needed property
`shape_observations.py`'s plain-torch functions already have. This module
therefore needs `python.sh`/a torch+rsl_rl-having interpreter to import
(this repo's Pi host has neither - see
`tests/test_mdp_shape_observations.py`'s own docstring for the established
"run tests via /home/saps/IsaacLab/_isaac_sim/python.sh" convention this
module's own tests follow), but never needs a live Isaac Sim/Kit process.

=====================================================================
IMITATION-LOSS FORMULATION CHOSEN (implementer's choice, per the spec's
own "implementing task's choice" instruction on this exact question)
=====================================================================

Multi-teacher DAgger (Ross, Gordon & Bagnell, "A Reduction of Imitation
Learning and Structured Prediction to No-Regret Online Learning," AISTATS
2011) with per-state expert ROUTING, deterministic MSE regression on the
actor's Gaussian mean action:

  - Two ROLLOUT environments run side by side, one per teacher's own
    single-shape env (d20's `FrankaDieLiftJointBigEnvCfg`, d12's
    `FrankaDieLiftJointD12BigEnvCfg`) - matching this task's own dispatch
    instruction to collect data "from each in its own single-shape env."
  - Within each environment, the ACTION ACTUALLY EXECUTED (which
    determines the next state visited) is a per-row Bernoulli(beta)
    mixture of the current student's own action and that env's own
    frozen teacher's action (`mix_actions`) - standard DAgger
    mixture-policy rollout, beta annealed from 1.0 (pure teacher, a
    behavior-cloning warm start) to 0.0 (pure student, full on-policy
    DAgger) via `dagger_beta_schedule`.
  - Every visited state, from EITHER environment, is relabeled with its
    OWN teacher's deterministic mean action (`MultiShapeTeacherRouter.
    relabel`) - the label always comes from the teacher whose shape
    matches that state's own shape-onehot observation feature (Task 1's
    schema), regardless of which action was actually executed to reach
    that state.
  - Both environments' relabeled (observation, teacher_action) pairs are
    POOLED and SHUFFLED together (`pool_and_shuffle`) before every
    supervised-regression minibatch, so a single gradient step's batch
    mixes both shapes (see "shape-randomized-per-episode design note"
    below).
  - Loss: plain MSE between the student's deterministic mean action
    (`ActorCritic.act_inference`) and the routed teacher's mean action
    (`behavior_cloning_loss`) - not a KL-regularized distillation loss.
    MSE-on-the-mean is the simpler of the two options the spec explicitly
    leaves open and is the standard continuous-control DAgger regression
    target (Ross et al. 2011's own formulation is loss-agnostic over the
    expert's actions; MSE-on-mean is the common, well-precedented
    instantiation for Gaussian-policy continuous control - `rsl_rl`'s own
    built-in `rsl_rl.algorithms.distillation.Distillation`, read directly
    on the desktop's installed `rsl_rl` package 2026-07-19, uses the same
    MSE-on-mean loss for its own single-teacher case).

Why NOT reuse `rsl_rl`'s own built-in `StudentTeacher` module /
`DistillationRunner` (`rsl_rl.modules.student_teacher.StudentTeacher`,
`rsl_rl.algorithms.distillation.Distillation`): that machinery is built
for exactly ONE frozen teacher - `StudentTeacher.load_state_dict` loads a
single actor's `actor.*`-prefixed weights into `self.teacher` and trains
a single `self.student` against it. This experiment has TWO frozen
teachers, each specialized on a disjoint shape population, so a
single-teacher class can't represent "the right label depends on which
shape this state's own episode is." Rather than force a two-teacher
problem through single-teacher machinery (e.g. running two separate
`StudentTeacher` distillation passes and somehow merging their students,
defeating the point of one unified shape-conditioned policy), this module
reuses `rsl_rl`'s own `ActorCritic`/`MLP` network classes directly (so
the distilled network topology is byte-identical to the existing PPO
stack - `tasks/franka/agents/rsl_rl_ppo_cfg.py`'s `FrankaLiftPPORunnerCfg`)
but implements its own thin multi-teacher routing + DAgger loop on top of
those classes, in plain PyTorch.

=====================================================================
"SHAPE-RANDOMIZED-PER-EPISODE" DESIGN NOTE (a real design choice, not
hidden)
=====================================================================

The dispatch brief for this task describes training "against the
shape-randomized-per-episode env ... shape assigned per-episode so the
unified policy actually has to condition on the observation rather than
memorize a single shape." This module does NOT introduce a new live env
cfg that resamples shape at each individual episode reset inside one
vectorized env instance. Two reasons:

  1. Isaac Lab's `MultiAssetSpawnerCfg(random_choice=True)` per-episode
     (as opposed to per-env-at-spawn-time) resampling semantics are an
     UNRESOLVED, previously-flagged risk elsewhere in this same plan
     (Task 3's own docstring in `tasks/franka/dice_lift_joint_env_cfg.py`:
     "the implementing task must verify whether Isaac Lab's
     `MultiAssetSpawnerCfg` actually supports per-*episode*-reset
     resampling or only per-env-at-spawn-time assignment"). Building a
     new mechanism here would either duplicate that unresolved
     investigation or silently assume an answer to it.
  2. It isn't actually needed to satisfy the stated goal. "Shape varies
     episode-to-episode, and the policy must condition on the
     observation to do well on both" is a property of the TRAINING DATA
     DISTRIBUTION the student's gradient updates are computed against,
     not a property that requires a single live env instance to
     internally vary shape. Running the two teachers' own existing
     single-shape envs side by side and pooling+shuffling their visited
     states before every regression step (`pool_and_shuffle`) produces
     the identical statistical training distribution - across the pooled
     stream, consecutive minibatches (and, in the aggregate, consecutive
     episodes) span both shapes, and a policy that ignores the
     shape-onehot feature and memorizes one shape's action mapping will
     do measurably worse on the other shape's states in the SAME batch.
     This sidesteps the open per-episode-resampling question entirely
     rather than resolving it - an explicit, flagged scope choice for
     this task, not an oversight.

If Task 5's real on-policy DAgger phase later needs genuinely-live
in-episode shape switching, that is a new, separate design question for
whoever runs Task 5 to raise, not something silently assumed resolved
here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable

import torch
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic

DEFAULT_D20_CHECKPOINT = (
    "gs://rl-manipulation-hks-runs/unified-multi-die-specialists/"
    "joint-die-big/seed123/2026-07-19_12-46-42/model_1499.pt"
)
DEFAULT_D12_CHECKPOINT = (
    "gs://rl-manipulation-hks-runs/unified-multi-die-specialists/"
    "joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt"
)

# franka_checkpoint_review.py --variant names for each teacher's own
# single-shape env - documentation/cross-reference only (scripts/
# distill_specialists.py's build_real_env imports the env cfg classes
# directly, not through that script).
TEACHER_ENV_VARIANTS = {"d20": "joint-die-big", "d12": "joint-die-d12-big"}

# tasks/franka/agents/rsl_rl_ppo_cfg.py's FrankaLiftPPORunnerCfg.policy -
# the student network is built with the IDENTICAL architecture so Task 6's
# later PPO fine-tune can load this checkpoint unchanged.
STUDENT_ACTOR_HIDDEN_DIMS = [256, 128, 64]
STUDENT_CRITIC_HIDDEN_DIMS = [256, 128, 64]
STUDENT_ACTIVATION = "elu"
STUDENT_INIT_NOISE_STD = 1.0

# Reference-only derivation of the current schema's shape-onehot offset
# (tasks/franka/lift_env_cfg.py's ObservationsCfg.PolicyCfg term order):
# joint_pos_rel(9) + joint_vel_rel(9) + object_position(3) +
# target_object_position(7) + last_action(8) = 36, then shape_class(4)
# starts at index 36, geometry_descriptor(1) at index 40. Verified against
# both real teacher checkpoints' own inferred obs_dim=41
# (`inspect_checkpoint_shapes`, this task's own dispatch verification
# step, 2026-07-19). Used only as scripts/distill_specialists.py's
# --dry-run default and this module's own unit tests' synthetic-schema
# default - the REAL (non-dry-run) pipeline derives this offset LIVE from
# the actual env's `observation_manager` (see
# `compute_shape_onehot_offset`), never from this hardcoded constant, so a
# future schema change can't silently desync the real run.
REFERENCE_SHAPE_ONEHOT_START = 36
REFERENCE_SHAPE_ONEHOT_DIM = 4
REFERENCE_OBS_DIM = 41
REFERENCE_NUM_ACTIONS = 8

DEFAULT_CHECKPOINT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "distill_checkpoint_cache"
)

_GCLOUD_FALLBACK = os.path.join(os.path.expanduser("~"), "google-cloud-sdk", "bin", "gcloud")


# =====================================================================
# Checkpoint loading / shape verification
# =====================================================================


def find_gcloud() -> str:
    """Same fallback convention as scripts/sync_run_to_gcs.py's own find_gcloud()."""
    found = shutil.which("gcloud")
    if found:
        return found
    if os.path.isfile(_GCLOUD_FALLBACK) and os.access(_GCLOUD_FALLBACK, os.X_OK):
        return _GCLOUD_FALLBACK
    raise FileNotFoundError(f"could not find 'gcloud' on PATH or at {_GCLOUD_FALLBACK}")


def _subprocess_env_for_gcloud() -> dict:
    """A real bug found and fixed while verifying this module's own
    --dry-run smoke test (2026-07-19): invoking `gcloud` via `subprocess`
    from a process launched through Isaac Sim's bundled Python
    (`/home/saps/IsaacLab/_isaac_sim/python.sh`) inherits that Python's own
    `PYTHONPATH` (pointed at Isaac Sim's bundled `kit/python/lib/python3.11`
    tree). `gcloud`'s own launcher script execs a *different* Python
    interpreter but still picks up this inherited `PYTHONPATH`, which
    imports Isaac's compiled `_sre`/`re` extension modules into an
    interpreter they weren't built for, crashing with `AssertionError: SRE
    module mismatch` before `gcloud` ever runs. `scripts/sync_run_to_gcs.py`
    never hits this because its own docstring already mandates running it
    under plain python3 (never Isaac's), sidestepping the issue rather than
    fixing it - this module can't make that same assumption, since its own
    CLI (`scripts/distill_specialists.py`) is designed to run under
    `_isaac_sim/python.sh` for both `--dry-run` and the real run. Fix:
    strip `PYTHONPATH`/`PYTHONHOME` from the child `gcloud` subprocess's own
    environment so it starts clean, regardless of which Python launched
    this module."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def download_checkpoint(path: str, cache_dir: str) -> str:
    """Returns a local filesystem path for `path`. Local paths pass through
    unchanged; `gs://...` URIs are downloaded once via `gcloud storage cp`
    into `cache_dir` and cached there for subsequent calls (keyed off the
    full gs:// path, so different runs/seeds/timestamps never collide)."""
    if not path.startswith("gs://"):
        return path
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, path.replace("gs://", "").replace("/", "_"))
    if os.path.isfile(local_path):
        return local_path
    subprocess.run([find_gcloud(), "storage", "cp", path, local_path], check=True, env=_subprocess_env_for_gcloud())
    return local_path


def inspect_checkpoint_shapes(checkpoint_path: str, cache_dir: str = DEFAULT_CHECKPOINT_CACHE_DIR) -> tuple[int, int]:
    """(obs_dim, num_actions) inferred directly from the checkpoint's own
    `actor.0.weight`/final-layer weight shapes - self-describing, makes no
    assumption about the exact observation-term composition (robust to a
    future schema change, unlike hardcoding 41/8)."""
    local_path = download_checkpoint(checkpoint_path, cache_dir)
    state = torch.load(local_path, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"]
    actor_weight_keys = sorted(
        (k for k in sd if k.startswith("actor.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    if not actor_weight_keys:
        raise ValueError(f"{checkpoint_path}: no 'actor.*.weight' keys found in model_state_dict")
    obs_dim = int(sd[actor_weight_keys[0]].shape[1])
    num_actions = int(sd[actor_weight_keys[-1]].shape[0])
    return obs_dim, num_actions


def load_frozen_teacher(
    checkpoint_path: str, obs_dim: int, num_actions: int, device: str = "cpu", cache_dir: str = DEFAULT_CHECKPOINT_CACHE_DIR
) -> ActorCritic:
    """Loads a frozen (eval-mode, no-grad) `rsl_rl.modules.ActorCritic`
    matching `FrankaLiftPPORunnerCfg`'s architecture from a real rsl_rl
    training checkpoint. `checkpoint_path` may be local or `gs://`."""
    local_path = download_checkpoint(checkpoint_path, cache_dir)
    state = torch.load(local_path, map_location=device, weights_only=False)
    dummy_obs = {"policy": torch.zeros(1, obs_dim, device=device)}
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    actor_critic = ActorCritic(
        dummy_obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=list(STUDENT_ACTOR_HIDDEN_DIMS),
        critic_hidden_dims=list(STUDENT_CRITIC_HIDDEN_DIMS),
        activation=STUDENT_ACTIVATION,
        init_noise_std=STUDENT_INIT_NOISE_STD,
    )
    actor_critic.load_state_dict(state["model_state_dict"])
    actor_critic.to(device)
    actor_critic.eval()
    for p in actor_critic.parameters():
        p.requires_grad_(False)
    return actor_critic


def build_student_actor_critic(obs_dim: int, num_actions: int, device: str = "cpu") -> ActorCritic:
    """Fresh, randomly-initialized ActorCritic with the SAME architecture as
    the frozen teachers (and as `FrankaLiftPPORunnerCfg`) - this is the
    unified policy this module trains. The critic is intentionally left
    untrained by this module's BC loss (see `save_student_checkpoint`'s own
    note): Task 6's PPO fine-tune trains it from scratch, standard practice
    for a distill-then-RL-finetune pipeline."""
    dummy_obs = {"policy": torch.zeros(1, obs_dim, device=device)}
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    return ActorCritic(
        dummy_obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=list(STUDENT_ACTOR_HIDDEN_DIMS),
        critic_hidden_dims=list(STUDENT_CRITIC_HIDDEN_DIMS),
        activation=STUDENT_ACTIVATION,
        init_noise_std=STUDENT_INIT_NOISE_STD,
    ).to(device)


def actor_inference_fn(actor_critic: ActorCritic) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wraps an ActorCritic's deterministic mean-action inference
    (`act_inference`) as a plain tensor->tensor callable (the interface
    every function below operates on) - used for both frozen teachers and
    the student."""

    def _fn(obs: torch.Tensor) -> torch.Tensor:
        return actor_critic.act_inference({"policy": obs})

    return _fn


# =====================================================================
# Multi-teacher routing
# =====================================================================


class MultiShapeTeacherRouter:
    """Relabels a batch of observations with each row's OWN shape's frozen
    teacher action, reading which shape each row is from the observation's
    own shape-onehot feature (Task 1's schema) rather than from any
    external per-row bookkeeping - so this works correctly even on a
    POOLED, SHUFFLED batch spanning multiple shapes (see module docstring's
    pool_and_shuffle design note)."""

    def __init__(
        self,
        teacher_action_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
        shape_classes: tuple[str, ...],
        shape_onehot_start: int,
        shape_onehot_dim: int = 4,
    ):
        if not teacher_action_fns:
            raise ValueError("teacher_action_fns must be non-empty")
        unknown = set(teacher_action_fns) - set(shape_classes)
        if unknown:
            raise ValueError(f"teacher_action_fns has shapes not in shape_classes: {sorted(unknown)}")
        self._teacher_action_fns = teacher_action_fns
        self._shape_classes = shape_classes
        self._start = shape_onehot_start
        self._dim = shape_onehot_dim

    def relabel(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[0] == 0:
            raise ValueError("cannot relabel an empty observation batch")
        onehot = obs[:, self._start : self._start + self._dim]
        shape_idx = onehot.argmax(dim=-1)
        out = None
        for i, shape in enumerate(self._shape_classes):
            mask = shape_idx == i
            if not bool(mask.any()):
                continue
            if shape not in self._teacher_action_fns:
                raise KeyError(
                    f"observation batch contains shape {shape!r} (row onehot argmax) but no teacher was "
                    f"registered for it (registered: {sorted(self._teacher_action_fns)})"
                )
            with torch.no_grad():
                action = self._teacher_action_fns[shape](obs[mask])
            if out is None:
                out = torch.zeros((obs.shape[0], action.shape[-1]), dtype=action.dtype, device=action.device)
            out[mask] = action
        return out


# =====================================================================
# DAgger mechanics: rollout collection, mixture policy, pooling, loss
# =====================================================================


def collect_rollout(env, action_fn: Callable[[torch.Tensor], torch.Tensor], num_steps: int, device: str = "cpu") -> torch.Tensor:
    """Steps `env` for `num_steps` control steps using
    `actions = action_fn(obs["policy"])`, returning every visited policy
    observation stacked as a single `(num_steps * num_envs, obs_dim)`
    tensor (steps are the outer dim; reshape to
    `(num_steps, num_envs, obs_dim)` if per-step/per-env structure is
    needed).

    `env` must expose the SAME minimal contract a real Isaac Lab env
    presents after `isaaclab_rl.rsl_rl.RslRlVecEnvWrapper` wraps it:
    `reset() -> obs | (obs, extras)` and `step(actions) -> (obs, reward,
    done, extras)`, where `obs` supports `obs["policy"] ->
    (num_envs, obs_dim) tensor` (a plain dict or a TensorDict both work -
    only `__getitem__` is used). This is intentionally identical to the
    real wrapper's own return contract so a stub test env exercises this
    function exactly as a real Isaac Sim env would, without ever launching
    Isaac Sim. No gradient tracking here (`torch.no_grad()`, not
    `torch.inference_mode()` - inference-mode tensors can't later be fed
    into an autograd-tracked op, which the BC regression step needs to do
    with these same stored observations)."""
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    collected = []
    with torch.no_grad():
        for _ in range(num_steps):
            policy_obs = obs["policy"].to(device)
            collected.append(policy_obs)
            actions = action_fn(policy_obs)
            step_result = env.step(actions)
            obs = step_result[0]
    return torch.cat(collected, dim=0)


def dagger_beta_schedule(iteration: int, num_iterations: int, beta_start: float = 1.0, beta_end: float = 0.0) -> float:
    """Linear anneal from `beta_start` (iteration 0) to `beta_end`
    (iteration `num_iterations - 1`) - the fraction of rollout steps whose
    EXECUTED action comes from the teacher rather than the student
    (Ross et al. 2011's mixture-policy DAgger)."""
    if num_iterations <= 1:
        return beta_end
    frac = min(max(iteration / (num_iterations - 1), 0.0), 1.0)
    return beta_start + frac * (beta_end - beta_start)


def mix_actions(
    student_actions: torch.Tensor, teacher_actions: torch.Tensor, beta: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Per-row Bernoulli(beta) selection of the teacher's action (True) vs.
    the student's own action (False) - the actual mixture-policy action
    EXECUTED in the env, i.e. which states get visited next. This is
    independent of the supervised-learning LABEL used for the BC loss,
    which always comes from the teacher regardless of beta (see
    `MultiShapeTeacherRouter.relabel`) - beta only controls exploration
    coverage, not label quality."""
    if student_actions.shape != teacher_actions.shape:
        raise ValueError(f"shape mismatch: student {tuple(student_actions.shape)} vs teacher {tuple(teacher_actions.shape)}")
    probs = torch.full((student_actions.shape[0],), float(beta), device=student_actions.device)
    use_teacher = torch.bernoulli(probs, generator=generator).bool()
    return torch.where(use_teacher.unsqueeze(-1), teacher_actions, student_actions)


def pool_and_shuffle(batches: list[torch.Tensor], generator: torch.Generator | None = None) -> torch.Tensor:
    """Concatenates observation batches from every shape's own env and
    shuffles the union along dim 0 - the mechanism that makes a single
    downstream regression minibatch mix shapes together (see module
    docstring's "shape-randomized-per-episode" design note: this is what
    stands in for a live per-episode-resampling env)."""
    pooled = torch.cat(batches, dim=0)
    perm = torch.randperm(pooled.shape[0], generator=generator, device="cpu")
    return pooled[perm.to(pooled.device)]


def behavior_cloning_loss(student_action_mean: torch.Tensor, teacher_action_mean: torch.Tensor) -> torch.Tensor:
    """MSE between the student's deterministic mean action and the routed
    teacher's mean action - see module docstring's imitation-loss section
    for why MSE-on-mean (not a KL-regularized distillation loss) was
    chosen."""
    return F.mse_loss(student_action_mean, teacher_action_mean)


def regress_on_pooled_batches(
    per_shape_batches: list[torch.Tensor],
    student: ActorCritic,
    router: MultiShapeTeacherRouter,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_epochs: int,
    generator: torch.Generator | None = None,
) -> float:
    """Pools+shuffles already-collected per-shape observation batches,
    relabels the pooled batch with each row's own teacher, and runs
    `num_epochs` passes of minibatch BC regression over it. Returns the mean
    loss across all regression steps. Extracted out of `run_dagger_iteration`
    (2026-07-19, Task 5's real-run bug fix - see that function's own updated
    docstring for why) so a real-run driver can call rollout collection and
    this regression step separately, with env open/close in between, without
    duplicating the regression logic or its own test coverage."""
    pooled_obs = pool_and_shuffle(per_shape_batches, generator=generator)
    teacher_labels = router.relabel(pooled_obs)

    losses = []
    n = pooled_obs.shape[0]
    for _epoch in range(num_epochs):
        perm = torch.randperm(n, generator=generator, device="cpu").to(pooled_obs.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_obs = pooled_obs[idx]
            batch_labels = teacher_labels[idx]
            student_mean = student.act_inference({"policy": batch_obs})
            loss = behavior_cloning_loss(student_mean, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return sum(losses) / len(losses)


def run_dagger_iteration(
    envs: dict[str, object],
    student: ActorCritic,
    router: MultiShapeTeacherRouter,
    optimizer: torch.optim.Optimizer,
    rollout_steps: int,
    batch_size: int,
    num_epochs: int,
    beta: float,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> float:
    """One full DAgger iteration: roll out the current beta-mixture policy
    in every shape's own env, relabel every visited state with its own
    teacher, pool+shuffle across shapes, and run `num_epochs` passes of
    minibatch BC regression over the pooled buffer. Returns the mean loss
    across all regression steps this iteration (for logging/plateau
    checks). This is the function this task's own unit tests exercise
    end-to-end against stub envs + stub/loaded teachers (which CAN stay
    open simultaneously, unlike a real Isaac Lab env - see
    tests/test_distillation_data_collection.py) and `--dry-run` exercises
    with the same stub envs.

    REAL-RUN CAVEAT, found under Task 5's first actual training run
    (2026-07-19, not caught by --dry-run's stub env, which has no notion of
    a simulation context): this function assumes BOTH envs in `envs` can be
    alive/open at once. A real Isaac Lab `ManagerBasedRLEnv` cannot -
    `SimulationContext` is a process-wide singleton and constructing a
    second `ManagerBasedRLEnv` while a first one is still open raises
    `RuntimeError: Simulation context already exists.` (confirmed by direct
    source read, isaaclab/envs/manager_based_env.py's `__init__`, and by
    hitting this for real on this task's first real dispatch). This
    function is therefore NOT used for the real (non-dry-run) run -
    `scripts/distill_specialists.py`'s real-run branch instead opens each
    shape's env one at a time (`collect_rollout`, then `env.close()`,
    before opening the next), collects both shapes' batches sequentially,
    and calls `regress_on_pooled_batches` (extracted from this function's
    own tail, unchanged logic) once both are collected and both envs are
    closed - achieving the identical statistical procedure (mixture-policy
    rollout against the CURRENT student in each shape's own env, both
    shapes' visited states pooled+shuffled before one regression step) with
    no envs ever open concurrently. This function itself is kept unchanged
    (still correct and tested for the stub/dry-run case, and still the
    simplest description of the intended mechanism) rather than deleted or
    forced into the sequential shape - `--dry-run`'s own stub envs have no
    simulation-context constraint, so exercising this exact function is
    still the right verification for that mode."""
    student_action_fn = actor_inference_fn(student)
    per_shape_batches = []
    for _shape, env in envs.items():

        def _mixture_action_fn(obs: torch.Tensor) -> torch.Tensor:
            student_actions = student_action_fn(obs)
            teacher_actions = router.relabel(obs)
            return mix_actions(student_actions, teacher_actions, beta, generator=generator)

        per_shape_batches.append(collect_rollout(env, _mixture_action_fn, rollout_steps, device=device))

    return regress_on_pooled_batches(per_shape_batches, student, router, optimizer, batch_size, num_epochs, generator)


# =====================================================================
# Live-env introspection (isaaclab-free - just reads attributes off a
# passed-in real env object; only ever called with a real Isaac Lab env,
# by scripts/distill_specialists.py's non-dry-run branch).
# =====================================================================


def compute_shape_onehot_offset(env) -> tuple[int, int]:
    """(shape_onehot_start, shape_onehot_dim) derived LIVE from a real
    env's own `observation_manager` term order/dims - never from
    REFERENCE_SHAPE_ONEHOT_START, so a future schema change can't silently
    desync this from reality. Requires a real Isaac Lab env (Task 5 only,
    via scripts/distill_specialists.py's build_real_env)."""
    manager = env.unwrapped.observation_manager
    term_names = manager.active_terms["policy"]
    term_dims = manager.group_obs_term_dim["policy"]
    offset = 0
    for name, dims in zip(term_names, term_dims):
        if name == "shape_class":
            return offset, int(dims[0])
        offset += int(dims[0])
    raise ValueError(f"'shape_class' term not found in policy observation group (terms: {term_names})")


# =====================================================================
# Checkpoint saving
# =====================================================================


def save_student_checkpoint(student: ActorCritic, output_path: str, iteration: int, extra_infos: dict | None = None) -> None:
    """Writes a checkpoint in the SAME `{"model_state_dict", "optimizer_state_dict",
    "iter", "infos"}` format `rsl_rl.runners.OnPolicyRunner.save()` uses
    (see `scripts/franka_checkpoint_review.py`'s own `runner.load(...)`
    call for the consumer side) so Task 6's PPO fine-tune can
    `runner.load()` this checkpoint directly. `optimizer_state_dict` is
    intentionally empty - Task 6's PPO fine-tune builds its own fresh PPO
    optimizer state, it doesn't resume this module's BC optimizer's Adam
    moments."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": {},
            "iter": iteration,
            "infos": {"source": "distill_specialists.py", **(extra_infos or {})},
        },
        output_path,
    )
