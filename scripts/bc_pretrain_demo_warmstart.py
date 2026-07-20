#!/usr/bin/env python
"""Task 2 of docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-
implementation.md: BC-pretrains a fresh student policy against Task 1's own
captured demonstration trajectories (`data/franka_demo_trajectories/{shape}/
seed{N}.pt`), replayed through the real target RL env, then saves a
checkpoint `scripts/train_franka.py --checkpoint --policy_only_checkpoint`
can resume for the PPO fine-tune (Task 3).

**Design choice: a new sibling CLI, not an extension of
`scripts/distill_specialists.py`** (documented per this task's own dispatch
instruction to make and record this call). `distill_specialists.py` is built
end-to-end around `tasks/franka/distillation.py`'s two-*live*-teacher DAgger
machinery (`MultiShapeTeacherRouter`, `dagger_beta_schedule`, `mix_actions`,
the beta-mixture rollout policy) - none of which apply here: H1 has exactly
one *fixed, already-recorded* demonstration source per shape, no live
teacher network to query or mix against, and no per-row routing decision to
make (every row in a given shape's replay batch gets that same shape's own
logged action, always). Forcing this through `distill_specialists.py`'s
router/mixture contract would mean either faking a "teacher" that just
replays a fixed action sequence (distorting that contract's own meaning) or
adding an if/else fork through code that assumes a live queryable policy
throughout. A new sibling script mirrors this project's own repeated
precedent for exactly this situation (`dice_pick_demo.py`'s Gate V reusing
Gate G by import rather than the reverse; `scripts/extract_demo_trajectory.py`
itself, Task 1's own new-sibling-script choice for the same reason) and
keeps `distill_specialists.py`'s own, separately-tested DAgger contract
untouched.

=====================================================================
REAL BUG found and fixed (2026-07-19, this task's own verification pass,
BEFORE ever attempting a real replay - found by direct calculation, not by
running it and observing corrupted data): episode-length mismatch between
Task 1's captured demonstrations and the real training env's default episode
cap
=====================================================================

`tasks/franka/lift_env_cfg.py`'s `FrankaLiftEnvCfg.__post_init__` sets
`episode_length_s = 5.0`, `decimation = 2`, `sim.dt = 0.01` - a 250-control-
step cap (`5.0 / (0.01 * 2)`), inherited unchanged by
`FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg` (no override). Task 1's own
captured trajectories log ONE joint-position/gripper target per RAW PHYSICS
STEP of `dice_pick_demo.py`'s own scripted pick sequence (module docstring
of `scripts/extract_demo_trajectory.py`) - up to ~850 entries per capture
(confirmed directly, `data/franka_demo_trajectories/d8/seed42.pt`:
`arm_joint_pos_target.shape == (848, 7)`). This module's own replay strategy
issues ONE captured entry per `env.step()` call (see
`build_replay_action_fn`/`replay_trajectory_to_paired_batch` below - holding
each entry's target for the env's own `decimation`-substep window is an
accepted, deliberate fidelity tradeoff, not itself a bug, see
`required_episode_length_s`'s own docstring), so replaying a 848-entry
capture consumes `848 * decimation(2) * sim.dt(0.01) = 16.96` sim-seconds -
over 3x the 5.0s cap. Left unfixed, `ManagerBasedRLEnv.step()`'s own internal
time-out auto-reset would fire partway through EVERY real replay, silently
resetting the scene to a fresh randomized state mid-trajectory while this
driver kept blindly feeding the ORIGINAL capture's later joint targets at
it - corrupting every (obs, action) pair collected after the reset point,
with no exception raised to flag it (a "success" data-flow that quietly
never terminated with real errors while inserting many mismatched labels
into a study population, the same measurement-class of bug this project has
hit and fixed with real (not merely typed/exit-code) verification discipline
elsewhere). Fixed by overriding `episode_length_s` upward, computed live
from the actual longest trajectory about to be replayed for that shape
(`required_episode_length_s`), applied ONLY to the throwaway env THIS
script builds for BC-pretrain data collection (`build_real_shape_env`) -
Task 3's real PPO fine-tune goes through `scripts/train_franka.py`'s own,
entirely separate env construction, using the standard 5.0s episode length
completely unaffected by this override.

=====================================================================
Known, deliberate property (per the plan's own Task 2 Files section, NOT a
bug this task introduced or is responsible for fixing): each captured
trajectory is replayed against a FRESHLY RESET env (`collect_rollout`'s own
`env.reset()` call, once per trajectory), whose randomized die/robot layout
is statistically independent of whatever specific layout
`dice_pick_demo.py`'s own capture run actually had. The captured ABSOLUTE
joint-position targets are replayed unchanged regardless - this is the
architecture the plan itself specifies ("reset the env, call
collect_rollout(...) with a scripted-replay action_fn"), not something this
task redesigns. Flagged here for visibility, not silently glossed over.
=====================================================================

Usage:

    # Mechanical smoke test (no Isaac Sim launch; stub trajectories + a
    # physics-free stub env stand in for the real captured data / real env):
    /home/saps/IsaacLab/_isaac_sim/python.sh scripts/bc_pretrain_demo_warmstart.py --dry-run

    # Real run (Task 2 Step 5): BC-pretrains both d8 and d10 against their
    # own 5 real captured trajectories, sequentially (one shape's env open/
    # replay/close before the next - this Isaac Lab installation cannot hold
    # two ManagerBasedRLEnvs open at once, tasks/franka/distillation.py's own
    # module docstring). Non-headless per CLAUDE.md's standing "the user
    # wants to watch" instruction.
    flock -o /tmp/rl_isaac_sim.lock -c \\
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/bc_pretrain_demo_warmstart.py"

    # Handoff smoke test (Task 2 Step 6) - NOT this script, a bounded
    # scripts/train_franka.py resume against the checkpoint this script just
    # wrote, e.g.:
    flock -o /tmp/rl_isaac_sim.lock -c \\
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \\
        --variant joint-die-d8-big --checkpoint logs/bc_pretrain_demo_warmstart/model_bc_d8.pt \\
        --policy_only_checkpoint --num_envs 64 --max_iterations <bc checkpoint's own saved iter + 5>"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Callable

# `tasks.franka...` must be importable regardless of cwd - same sys.path
# convention as scripts/franka_checkpoint_review.py / scripts/distill_specialists.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

import torch  # noqa: E402 - importable without an AppLauncher (see tasks/franka/distillation.py's own docstring)

from tasks.franka.demo_action_mapping import gripper_target_to_raw_action, joint_pos_to_raw_action  # noqa: E402
from tasks.franka.distillation import (  # noqa: E402
    build_student_actor_critic,
    collect_rollout,
    save_student_checkpoint,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAJECTORY_DIR = os.path.join(REPO_ROOT, "data", "franka_demo_trajectories")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "logs", "bc_pretrain_demo_warmstart")


# =====================================================================
# Pure-torch mechanics (NO isaaclab/pxr imports anywhere in this section) -
# unit-tested directly, no Isaac Sim launch needed
# (tests/test_bc_pretrain_demo_warmstart.py).
# =====================================================================


def required_episode_length_s(num_steps: int, decimation: int, sim_dt: float, safety_margin: float = 1.2) -> float:
    """Minimum `episode_length_s` a training env must be built with to
    replay a `num_steps`-long captured trajectory (Task 1's own per-physics-
    step log) through it at one captured entry per `env.step()` call,
    without the env's own internal time-out auto-reset firing mid-replay -
    see this module's own docstring's "REAL BUG found and fixed" section.
    `num_steps * decimation * sim_dt` is the exact sim-seconds one full
    replay consumes (each `env.step()` call advances physics by
    `decimation` substeps of `sim_dt` each, using the SAME held target for
    all of them - holding a target across a `decimation`-substep window
    instead of updating every raw substep is an accepted, deliberate
    approximation for this BC-pretrain use, not itself a bug: the captured
    targets change smoothly/incrementally step to step, per
    `dice_pick_demo.py`'s own interpolated `_step_toward` controller, so a
    persistent PD setpoint held slightly longer per waypoint converges to
    essentially the same physical path). `safety_margin` (default 1.2 = 20%
    headroom) guards against the exact-equality boundary case."""
    if num_steps <= 0:
        raise ValueError(f"required_episode_length_s: num_steps must be positive, got {num_steps}")
    return num_steps * decimation * sim_dt * safety_margin


def demo_trajectory_to_raw_actions(trajectory: dict, default_joint_pos: torch.Tensor, arm_scale: float = 0.5) -> torch.Tensor:
    """Precomputes the full `(num_steps, 8)` raw-action tensor for one
    captured demonstration trajectory dict
    (`scripts/extract_demo_trajectory.py`'s own
    `{"arm_joint_pos_target": (num_steps, 7), "gripper_target": (num_steps,
    2), ...}` schema), via `tasks/franka/demo_action_mapping.py`'s
    closed-form conversions - vectorized once over the whole trajectory
    (both conversions are pure elementwise/reduction ops with no per-step
    state), rather than recomputed inside a per-step replay loop. Column
    order [0:7] = arm raw actions, [7:8] = gripper raw action, matching
    `tasks/franka/lift_env_cfg.py`'s `ActionsCfg` field declaration order
    (`arm_action` then `gripper_action` - confirmed by direct read,
    `dice_lift_joint_env_cfg.py`'s `FrankaDieLiftJointEnvCfg.__post_init__`
    overrides only `arm_action`, leaving `gripper_action` declared second,
    unchanged).

    REAL BUG found and fixed (2026-07-19, this task's own first real-GPU
    dispatch, never caught by the CPU-only unit tests/--dry-run): captured
    trajectories are loaded via `torch.load(path, map_location="cpu")`
    (`scripts/bc_pretrain_demo_warmstart.py`'s own real-run branch), so
    `trajectory["arm_joint_pos_target"]` stays on CPU, while callers (this
    module's own `bc_pretrain_shape`) read `default_joint_pos` LIVE off the
    real env (`get_default_arm_joint_pos`), which lives on the env's own
    `cuda` device. Subtracting a cuda tensor from a cpu tensor raises
    `RuntimeError: Expected all tensors to be on the same device`. Fixed
    here (not by pushing a `.to(device)` onto every caller) by aligning
    `default_joint_pos` to `arm_joint_pos_target`'s own device before the
    subtraction - this function's own output is still moved to the caller's
    target `device` afterward by `bc_pretrain_shape`, so this fix only
    changes where the intermediate compute happens (CPU, matching the
    trajectory's own natural device), not the final result's device."""
    arm_target = trajectory["arm_joint_pos_target"]
    gripper_target = trajectory["gripper_target"]
    raw_arm = joint_pos_to_raw_action(arm_target, default_joint_pos.to(arm_target.device), scale=arm_scale)
    raw_gripper = gripper_target_to_raw_action(gripper_target)
    return torch.cat([raw_arm, raw_gripper], dim=-1)


def build_replay_action_fn(raw_actions: torch.Tensor, device: str = "cpu") -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a `collect_rollout`-compatible `action_fn(obs) -> action`
    closure that ignores `obs`, looks up `raw_actions`' current-step row
    (broadcast to `obs.shape[0]` rows), and increments an internal step
    counter each call - a scripted, open-loop replay of an
    already-converted demonstration trajectory. Raises `IndexError` if
    called past `raw_actions`' own length (a caller bug - `collect_rollout`
    should always be given `num_steps == raw_actions.shape[0]`, see
    `replay_trajectory_to_paired_batch`)."""

    step = {"i": 0}

    def _fn(obs: torch.Tensor) -> torch.Tensor:
        i = step["i"]
        if i >= raw_actions.shape[0]:
            raise IndexError(f"build_replay_action_fn: called for step {i} but raw_actions only has {raw_actions.shape[0]} rows")
        row = raw_actions[i].to(device)
        step["i"] += 1
        return row.unsqueeze(0).expand(obs.shape[0], -1)

    return _fn


def replay_trajectory_to_paired_batch(env, raw_actions: torch.Tensor, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Replays one already-converted demonstration trajectory
    (`raw_actions`, `demo_trajectory_to_raw_actions`'s own output) through
    `env` via `collect_rollout`, returning the paired `(obs, actions)`
    tensors for this trajectory alone. `collect_rollout` itself only
    returns the visited OBSERVATIONS, not the actions that produced them
    (see its own docstring) - since `raw_actions` is fully known ahead of
    time (a deterministic, pre-computed lookup, not a function of `obs`),
    the paired action for observation row `i` is exactly `raw_actions[i //
    env.num_envs]` (`collect_rollout`'s own outer loop is "steps", inner
    dim "envs" - see its docstring's `(num_steps * num_envs, obs_dim)`
    return shape), reconstructed here via `repeat_interleave` rather than a
    side-effecting log."""
    action_fn = build_replay_action_fn(raw_actions, device=device)
    obs = collect_rollout(env, action_fn, num_steps=raw_actions.shape[0], device=device)
    actions = raw_actions.to(device).repeat_interleave(env.num_envs, dim=0)
    return obs, actions


def pool_trajectory_batches(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenates multiple trajectories' own paired `(obs, actions)`
    batches (Task 2's own "pool the 5 replays' paired tensors" step) - a
    plain row-order concat, not a shuffle (unlike
    `tasks/franka/distillation.py`'s `pool_and_shuffle`):
    `regress_on_paired_batches` already reshuffles every epoch internally,
    so an external pre-shuffle here would be redundant, not incorrect."""
    if not pairs:
        raise ValueError("pool_trajectory_batches: pairs must be non-empty")
    obs = torch.cat([p[0] for p in pairs], dim=0)
    actions = torch.cat([p[1] for p in pairs], dim=0)
    return obs, actions


def get_default_arm_joint_pos(env) -> torch.Tensor:
    """Reads the LIVE default arm-joint-pos offset directly off `env`'s own
    robot articulation (`panda_joint.*`, the same 7-joint regex
    `FrankaDieLiftJoint*EnvCfg`'s own `arm_action` term uses) - Task 1's own
    module docstring requirement that this NOT be reused from a captured
    trajectory's own recorded `default_joint_pos` field (that field came
    from a DIFFERENT scene instance - `scripts/extract_demo_trajectory.py`'s
    own docstring flags this explicitly). Duck-typed against
    `env.unwrapped.scene["robot"]` exposing `.find_joints(name_keys) ->
    (indices, names)` and `.data.default_joint_pos -> (num_envs,
    num_joints) tensor` - the real `isaaclab.assets.Articulation` API
    (confirmed by direct source read,
    `isaaclab/assets/articulation/articulation.py:244`; no isaaclab import
    needed HERE, this function only calls methods on whatever object `env`
    already is), matched exactly by this module's own test/dry-run stub so
    this function is exercised identically to how it's used for real."""
    robot = env.unwrapped.scene["robot"]
    joint_ids, _joint_names = robot.find_joints(["panda_joint.*"])
    return robot.data.default_joint_pos[0, joint_ids]


def bc_pretrain_until_plateau(
    obs: torch.Tensor,
    actions: torch.Tensor,
    student,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    epochs_per_round: int,
    max_rounds: int,
    plateau_window: int = 5,
    plateau_rel_tol: float = 0.02,
    generator: torch.Generator | None = None,
) -> tuple[list[float], bool]:
    """Repeatedly calls `tasks.franka.distillation.regress_on_paired_batches`
    (each call = `epochs_per_round` epochs over the full pooled `(obs,
    actions)` dataset), logging the mean loss per round, until either
    `max_rounds` is reached or the loss has PLATEAUED: once at least
    `plateau_window` rounds have run, if the most recent round's loss is no
    more than `plateau_rel_tol` (relative) below the loss from
    `plateau_window` rounds earlier, stop early. Mirrors the prior
    experiment's Task 4 `--dry-run` precedent of logging per-round loss for
    the caller to inspect/report
    (`docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-
    implementation.md` Task 2's own "stop once it plateaus" step) - the
    returned `loss_history` is exactly what a real run reports/logs; this
    function's own automated stop is a bounded-runtime mechanism layered on
    top of that, not a replacement for reviewing the curve. Returns
    `(loss_history, stopped_early)`."""
    from tasks.franka.distillation import regress_on_paired_batches

    if max_rounds < 1:
        raise ValueError(f"bc_pretrain_until_plateau: max_rounds must be >= 1, got {max_rounds}")

    loss_history: list[float] = []
    for round_idx in range(max_rounds):
        loss = regress_on_paired_batches(obs, actions, student, optimizer, batch_size, epochs_per_round, generator=generator)
        loss_history.append(loss)
        print(f"bc_pretrain_until_plateau: round {round_idx + 1}/{max_rounds} mean_loss={loss:.6f}", flush=True)
        if len(loss_history) > plateau_window:
            prior = loss_history[-plateau_window - 1]
            current = loss_history[-1]
            if prior <= 0:
                continue
            rel_improvement = (prior - current) / prior
            if rel_improvement < plateau_rel_tol:
                return loss_history, True
    return loss_history, False


def bc_pretrain_shape(
    env,
    trajectories: list[dict],
    device: str = "cpu",
    batch_size: int = 256,
    epochs_per_round: int = 4,
    max_rounds: int = 50,
    plateau_window: int = 5,
    plateau_rel_tol: float = 0.02,
    learning_rate: float = 1.0e-3,
    arm_scale: float = 0.5,
    generator: torch.Generator | None = None,
):
    """Full per-shape driver (Task 2's own Files section): replays every
    captured trajectory in `trajectories` through `env` (already built,
    real or stub, consistent `num_envs` across all of them), pools the
    resulting paired `(obs, action)` data, builds a fresh student, and
    BC-regresses it to a loss plateau. `env`'s own default arm-joint-pos
    offset is read LIVE via `get_default_arm_joint_pos` (never reused from
    any trajectory's own recorded field - see that function's own
    docstring). `obs_dim`/`num_actions` are inferred directly from the
    pooled data's own shape (self-describing, same philosophy
    `tasks/franka/distillation.py`'s `inspect_checkpoint_shapes` already
    uses) rather than passed in or assumed from a reference constant.
    Returns `(student, loss_history, stopped_early)`."""
    import time as _time

    default_joint_pos = get_default_arm_joint_pos(env).to(device)
    pairs = []
    for traj_idx, trajectory in enumerate(trajectories):
        num_steps = trajectory["arm_joint_pos_target"].shape[0]
        print(f"bc_pretrain_shape: replaying trajectory {traj_idx + 1}/{len(trajectories)} ({num_steps} steps)...", flush=True)
        _start = _time.monotonic()
        raw_actions = demo_trajectory_to_raw_actions(trajectory, default_joint_pos, arm_scale=arm_scale).to(device)
        pairs.append(replay_trajectory_to_paired_batch(env, raw_actions, device=device))
        print(f"bc_pretrain_shape: trajectory {traj_idx + 1}/{len(trajectories)} replayed in {_time.monotonic() - _start:.1f}s", flush=True)
    pooled_obs, pooled_actions = pool_trajectory_batches(pairs)

    obs_dim = pooled_obs.shape[-1]
    num_actions = pooled_actions.shape[-1]
    student = build_student_actor_critic(obs_dim, num_actions, device)
    optimizer = torch.optim.Adam(student.actor.parameters(), lr=learning_rate)

    loss_history, stopped_early = bc_pretrain_until_plateau(
        pooled_obs,
        pooled_actions,
        student,
        optimizer,
        batch_size,
        epochs_per_round,
        max_rounds,
        plateau_window=plateau_window,
        plateau_rel_tol=plateau_rel_tol,
        generator=generator,
    )
    return student, loss_history, stopped_early


# =====================================================================
# Isaac-Lab-touching env construction (only called for a real, non-dry-run
# run - every isaaclab import lives inside this function, never at module
# level, matching scripts/distill_specialists.py's own established
# convention so --dry-run/--help never trigger them).
# =====================================================================


def build_real_shape_env(shape: str, num_envs: int, device: str, max_trajectory_num_steps: int | None = None):
    """Constructs the real target `FrankaDieLiftJointD8BigEnvCfg`/
    `...D10BigEnvCfg` env, wrapped exactly like
    `scripts/distill_specialists.py`'s own `build_real_mixed_env` wraps its
    env for `collect_rollout`'s contract. If `max_trajectory_num_steps` is
    given, `episode_length_s` is overridden upward (BEFORE constructing
    `ManagerBasedRLEnv` - the same "mutate the cfg pre-construction" pattern
    `scripts/_diag_d8d10_48mm_grasp_reverify.py`'s own `override_die_scale`
    already established) to comfortably exceed the longest trajectory about
    to be replayed - see this module's own docstring's "REAL BUG found and
    fixed" section and `required_episode_length_s` for why. This ONLY
    affects THIS throwaway BC-pretrain-data-collection env - Task 3's real
    PPO fine-tune goes through `scripts/train_franka.py`'s own, entirely
    separate env construction, unaffected."""
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    if shape == "d8":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8BigEnvCfg as EnvCfgCls
    elif shape == "d10":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD10BigEnvCfg as EnvCfgCls
    else:
        raise ValueError(f"build_real_shape_env: unknown shape {shape!r} (expected 'd8' or 'd10')")

    env_cfg = EnvCfgCls()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    if max_trajectory_num_steps is not None:
        needed = required_episode_length_s(max_trajectory_num_steps, env_cfg.decimation, env_cfg.sim.dt)
        if needed > env_cfg.episode_length_s:
            print(
                f"build_real_shape_env: overriding episode_length_s {env_cfg.episode_length_s}s -> {needed:.2f}s "
                f"so a {max_trajectory_num_steps}-step captured trajectory can replay without a mid-replay time-out reset"
            )
            env_cfg.episode_length_s = needed
    env = ManagerBasedRLEnv(cfg=env_cfg)
    return RslRlVecEnvWrapper(env, clip_actions=None)


# =====================================================================
# --dry-run stub trajectories/env (mirrors scripts/distill_specialists.py's
# own `_SyntheticDieEnv` precedent) - physics-free, no Isaac Sim launch.
# =====================================================================


class _StubRobot:
    """Minimal stand-in for `isaaclab.assets.Articulation`, satisfying only
    `get_default_arm_joint_pos`'s own duck-typed contract."""

    class _Data:
        pass

    def __init__(self, num_envs: int, num_joints: int, device: str):
        self.data = _StubRobot._Data()
        self.data.default_joint_pos = torch.zeros(num_envs, num_joints, device=device)

    def find_joints(self, _name_keys):
        return list(range(7)), [f"panda_joint{i + 1}" for i in range(7)]


class _StubReplayEnv:
    """Minimal stand-in for a real `RslRlVecEnvWrapper`-wrapped Franka
    die-lift env, satisfying `collect_rollout`'s contract PLUS
    `get_default_arm_joint_pos`'s own duck-typed
    `.unwrapped.scene["robot"].find_joints`/`.data.default_joint_pos`
    contract - used ONLY by `--dry-run`."""

    def __init__(self, num_envs: int, obs_dim: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self._state = torch.zeros(num_envs, obs_dim, device=device)
        self._robot = _StubRobot(num_envs, num_joints=9, device=device)
        self.scene = {"robot": self._robot}

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        return {"policy": self._state.clone()}

    def reset(self):
        self._state.zero_()
        return self._obs()

    def step(self, actions):
        self._state = (self._state + 0.001 * actions.mean(dim=-1, keepdim=True).to(self.device)).clamp(-10.0, 10.0)
        reward = torch.zeros(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._obs(), reward, done, {}


def _make_stub_trajectory(seed: int, num_steps: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    arm = torch.randn(num_steps, 7, generator=gen) * 0.1
    half = num_steps // 2
    gripper = torch.cat(
        [
            torch.full((half, 2), 0.04),
            torch.full((num_steps - half, 2), 0.0),
        ],
        dim=0,
    )
    return {"arm_joint_pos_target": arm, "gripper_target": gripper}


# =====================================================================
# CLI
# =====================================================================


def build_arg_parser(app_launcher_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BC-pretrain a fresh student policy from Task 1's captured demonstration trajectories, per shape."
    )
    parser.add_argument("--shapes", type=str, default="d8,d10", help="Comma-separated shapes to BC-pretrain (default both).")
    parser.add_argument("--trajectory-dir", type=str, default=DEFAULT_TRAJECTORY_DIR, help="Root dir of captured trajectories (data/franka_demo_trajectories).")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to write BC-pretrained checkpoint(s) to.")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel envs for the replay env - 1 (deterministic open-loop tracking), per the plan.")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for each BC regression round.")
    parser.add_argument("--epochs-per-round", type=int, default=4, help="Epochs over the pooled dataset per plateau-check round.")
    parser.add_argument("--max-rounds", type=int, default=50, help="Upper bound on plateau-check rounds before stopping unconditionally.")
    parser.add_argument("--plateau-window", type=int, default=5, help="Rounds back to compare against for the plateau check.")
    parser.add_argument("--plateau-rel-tol", type=float, default=0.02, help="Relative loss improvement below which training is considered plateaued.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Adam learning rate for the student actor's BC regression.")
    parser.add_argument("--arm-scale", type=float, default=0.5, help="JointPositionAction's own scale for the arm - FrankaDieLiftJointD8BigEnvCfg/...D10BigEnvCfg's own arm_action cfg value.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for student init + minibatch shuffling.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mechanical smoke test: stub trajectories + a physics-free stub env stand in for real captured data/env - no Isaac Sim launch.",
    )
    parser.add_argument("--dry-run-num-steps", type=int, default=12, help="Steps per stub trajectory under --dry-run (kept tiny).")
    parser.add_argument("--dry-run-num-trajectories", type=int, default=3, help="Stub trajectories per shape under --dry-run (kept tiny).")
    parser.add_argument("--dry-run-obs-dim", type=int, default=41, help="Stub env observation dim under --dry-run (matches the real 41-dim schema).")
    parser.add_argument("--dry-run-max-rounds", type=int, default=5, help="Plateau-check round cap under --dry-run (kept tiny).")
    app_launcher_cls.add_app_launcher_args(parser)
    return parser


def main(args_cli: argparse.Namespace) -> None:
    torch.manual_seed(args_cli.seed)
    generator = torch.Generator(device="cpu").manual_seed(args_cli.seed)
    shapes = [s.strip() for s in args_cli.shapes.split(",") if s.strip()]
    os.makedirs(args_cli.output_dir, exist_ok=True)

    if args_cli.dry_run:
        device = "cpu"
        for shape in shapes:
            print(f"[dry-run] shape={shape}: building stub trajectories + stub env (no Isaac Sim launch)...")
            trajectories = [_make_stub_trajectory(seed, args_cli.dry_run_num_steps) for seed in range(args_cli.dry_run_num_trajectories)]
            env = _StubReplayEnv(num_envs=args_cli.num_envs, obs_dim=args_cli.dry_run_obs_dim, device=device)
            student, loss_history, stopped_early = bc_pretrain_shape(
                env,
                trajectories,
                device=device,
                batch_size=args_cli.batch_size,
                epochs_per_round=args_cli.epochs_per_round,
                max_rounds=args_cli.dry_run_max_rounds,
                plateau_window=args_cli.plateau_window,
                plateau_rel_tol=args_cli.plateau_rel_tol,
                learning_rate=args_cli.learning_rate,
                arm_scale=args_cli.arm_scale,
                generator=generator,
            )
            print(f"[dry-run] shape={shape}: rounds={len(loss_history)} stopped_early={stopped_early} losses={loss_history}")
            out_path = os.path.join(args_cli.output_dir, f"bc_pretrain_dryrun_{shape}.pt")
            save_student_checkpoint(student, out_path, len(loss_history) - 1, {"dry_run": True, "shape": shape})
            print(f"[dry-run] shape={shape}: OK - wrote smoke-test checkpoint to {out_path}")
        return

    # Real run (Task 2 Step 5).
    device = args_cli.device
    for shape in shapes:
        traj_dir = os.path.join(args_cli.trajectory_dir, shape)
        traj_paths = sorted(glob.glob(os.path.join(traj_dir, "seed*.pt")))
        if not traj_paths:
            raise FileNotFoundError(f"no captured trajectories found under {traj_dir}")
        print(f"shape={shape}: loading {len(traj_paths)} captured trajectories: {traj_paths}")
        trajectories = [torch.load(p, map_location="cpu", weights_only=False) for p in traj_paths]
        for path, trajectory in zip(traj_paths, trajectories):
            if not trajectory.get("pass", False):
                raise ValueError(
                    f"{path}: captured trajectory has pass=False - not a valid demonstration "
                    "(Task 1's own drop-and-replace instruction; this file should never have been kept)"
                )
        max_num_steps = max(t["arm_joint_pos_target"].shape[0] for t in trajectories)

        env = build_real_shape_env(shape, args_cli.num_envs, device, max_trajectory_num_steps=max_num_steps)
        try:
            student, loss_history, stopped_early = bc_pretrain_shape(
                env,
                trajectories,
                device=device,
                batch_size=args_cli.batch_size,
                epochs_per_round=args_cli.epochs_per_round,
                max_rounds=args_cli.max_rounds,
                plateau_window=args_cli.plateau_window,
                plateau_rel_tol=args_cli.plateau_rel_tol,
                learning_rate=args_cli.learning_rate,
                arm_scale=args_cli.arm_scale,
                generator=generator,
            )
        finally:
            env.close()

        print(f"shape={shape}: BC-pretrain finished after {len(loss_history)} round(s) (stopped_early={stopped_early})")
        print(f"shape={shape}: loss curve: {loss_history}")
        out_path = os.path.join(args_cli.output_dir, f"model_bc_{shape}.pt")
        save_student_checkpoint(
            student,
            out_path,
            len(loss_history) - 1,
            {"source": "bc_pretrain_demo_warmstart.py", "shape": shape, "num_trajectories": len(trajectories)},
        )
        print(f"shape={shape}: saved BC-pretrained checkpoint to {out_path}")


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    parser = build_arg_parser(AppLauncher)
    args_cli = parser.parse_args()

    simulation_app = None
    if not args_cli.dry_run:
        # Real run only: actually boot Isaac Sim. --dry-run/--help never
        # reach this branch, matching scripts/distill_specialists.py's own
        # established pattern.
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    main(args_cli)

    if simulation_app is not None:
        simulation_app.close()
