#!/usr/bin/env python
"""Distill frozen per-shape Franka die-lift specialists into one unified
policy - CLI entry point.

Task 4 of `docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
distillation.md` ("Distillation pipeline (local, no GPU training yet)").

Task 4's own scope was the pipeline itself, verified mechanically (unit
tests in `tests/test_distillation_data_collection.py` + `--dry-run`
below), not a real training run. Task 5 (BACKLOG.md's 2026-07-19
controller decision "(b) single mixed-population env") runs this script
for real, without `--dry-run`, against a single
`FrankaDieLiftJointD12D20MixedEnvCfg` env (see `build_real_mixed_env`
below) instead of Task 5's own original, blocked, two-single-shape-envs
design - see that class's own docstring
(`tasks/franka/dice_lift_joint_env_cfg.py`) and BACKLOG.md's BLOCKED entry
for the full architectural trace.

Scope (2026-07-19 narrowing, `BACKLOG.md`'s "Task 4 scope decision: narrow
to d12+d20, defer d8/d10" entry + its "re-audit" follow-up correction):
exactly TWO frozen teacher specialists, not the plan's originally-envisioned
3-4 - d8/d10 are confirmed null across two independent size regimes and are
out of scope here entirely (a separate, deferred research question, not
silently dropped). The two in-scope teachers, both independently verified
loadable and shape-compatible before this script was designed (2026-07-19,
this task's own dispatch verification step - both carry a 41-dim
observation and an 8-dim action, loaded cleanly into a real
`rsl_rl.modules.ActorCritic` matching `FrankaLiftPPORunnerCfg`'s
architecture):

  d20: gs://rl-manipulation-hks-runs/unified-multi-die-specialists/
       joint-die-big/seed123/2026-07-19_12-46-42/model_1499.pt
       (trained against tasks.franka.dice_lift_joint_env_cfg.FrankaDieLiftJointBigEnvCfg;
       8/8 sustained lift, undiluted 48mm)
  d12: gs://rl-manipulation-hks-runs/unified-multi-die-specialists/
       joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt
       (trained against tasks.franka.dice_lift_joint_env_cfg.FrankaDieLiftJointD12BigEnvCfg;
       8/8 sustained lift once correctly measured, ROADMAP.md's
       "Task 3.5 re-audit" entry, 2026-07-19)

For the imitation-loss formulation chosen (multi-teacher DAgger + per-state
expert routing + MSE-on-mean regression) and the "shape-randomized-per-
episode" design note, see `tasks/franka/distillation.py`'s own module
docstring - the actual DAgger/BC mechanics live there, unchanged by Task 5
(only WHERE the pooled batch's two shapes come from changed - one mixed env
instead of two side-by-side single-shape envs; `regress_on_pooled_batches`
itself doesn't care whether its input list has one already-mixed batch or
several single-shape ones). This file is deliberately a thin argparse/
AppLauncher/real-env-construction wrapper around it, kept separate so
`tasks/franka/distillation.py` stays plain-importable (no argparse-at-
import-time, no `isaaclab.app.AppLauncher` needed at all) and therefore
unit-testable via `tests/test_distillation_data_collection.py` without
Isaac Sim.

Usage:

    # Mechanical smoke test (no Isaac Sim launch; real checkpoint
    # download+load+shape-check, real student ActorCritic, but stub
    # synthetic envs stand in for the real Isaac Lab single-shape envs):
    /home/saps/IsaacLab/_isaac_sim/python.sh scripts/distill_specialists.py --dry-run

    # Real run (Task 5): launches Isaac Sim against the single mixed env.
    # Non-headless per CLAUDE.md's standing "run non-headless, the user
    # wants to watch" instruction (drop --headless unless running on a
    # headless cloud box, in which case add it back per
    # docs/cloud/dispatch-checklist.md).
    flock -o /tmp/rl_isaac_sim.lock -c \\
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/distill_specialists.py \\
        --num-iterations 1500"
"""

from __future__ import annotations

import argparse
import os
import sys

# `tasks.franka...` must be importable regardless of cwd - same sys.path
# convention as scripts/franka_checkpoint_review.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

import torch  # noqa: E402 - importable without an AppLauncher (see tasks/franka/distillation.py's own docstring)

from tasks.franka.distillation import (  # noqa: E402
    DEFAULT_CHECKPOINT_CACHE_DIR,
    DEFAULT_D12_CHECKPOINT,
    DEFAULT_D20_CHECKPOINT,
    REFERENCE_NUM_ACTIONS,
    REFERENCE_OBS_DIM,
    MultiShapeTeacherRouter,
    actor_inference_fn,
    build_student_actor_critic,
    collect_rollout,
    compute_shape_onehot_offset,
    dagger_beta_schedule,
    inspect_checkpoint_shapes,
    load_frozen_teacher,
    mix_actions,
    regress_on_pooled_batches,
    run_dagger_iteration,
    save_student_checkpoint,
)
from tasks.franka.shape_observations import SHAPE_CLASSES  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "distill_franka")


class _SyntheticDieEnv:
    """Minimal stand-in for a real Isaac Lab single-shape env, satisfying
    exactly the contract `tasks.franka.distillation.collect_rollout` needs
    (`reset()`/`step()` returning dict-like observations with a `"policy"`
    key). Used ONLY by `--dry-run` (this task's own mechanical smoke test) -
    NOT used by the real Task 5 run, which calls `build_real_env` instead.
    Physics-free: observations are a fixed shape-onehot block plus an
    unconstrained random-walk over the remaining dims, just enough to
    exercise the pipeline's data-flow/shapes end-to-end."""

    def __init__(self, shape_index: int, num_shapes: int, num_envs: int, obs_dim: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self._shape_onehot = torch.zeros(num_shapes, device=device)
        self._shape_onehot[shape_index] = 1.0
        self._non_shape_dim = obs_dim - num_shapes
        self._state = torch.zeros(num_envs, self._non_shape_dim, device=device)

    def _obs(self) -> dict:
        onehot_block = self._shape_onehot.unsqueeze(0).expand(self.num_envs, -1)
        return {"policy": torch.cat([self._state, onehot_block], dim=-1)}

    def reset(self):
        self._state.zero_()
        return self._obs()

    def step(self, actions):
        # Real bug found and fixed while running this task's own --dry-run
        # smoke test (2026-07-19): the real env's action dim (8, joint
        # controls) has no relationship to its observation dim (41) minus
        # the shape-onehot block (39) - slicing `actions` to
        # `self._non_shape_dim` columns crashes whenever num_actions !=
        # non_shape_dim (true for the real schema: 8 != 39). This stub is
        # physics-free and doesn't need per-dim action->state coupling
        # anyway, so the fix broadcasts a single per-env scalar summary of
        # the action (its mean) across every state column instead -
        # dimension-agnostic regardless of how obs_dim/num_actions relate,
        # while still making state evolution depend on the action actually
        # taken (exercising the same "actions influence visited states"
        # data-flow property the real env has, just without a real physics
        # coupling).
        self._state = (self._state + 0.01 * actions.mean(dim=-1, keepdim=True).to(self.device)).clamp(-10.0, 10.0)
        reward = torch.zeros(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._obs(), reward, done, {}


def build_arg_parser(app_launcher_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill frozen per-shape Franka die-lift specialists into one unified policy.")
    parser.add_argument("--d20-checkpoint", type=str, default=DEFAULT_D20_CHECKPOINT, help="d20 frozen teacher checkpoint (local path or gs:// URI).")
    parser.add_argument("--d12-checkpoint", type=str, default=DEFAULT_D12_CHECKPOINT, help="d12 frozen teacher checkpoint (local path or gs:// URI).")
    parser.add_argument("--num-iterations", type=int, default=200, help="Number of DAgger iterations (rollout + BC regression steps).")
    parser.add_argument("--rollout-steps", type=int, default=24, help="Control steps collected per env per iteration (matches FrankaLiftPPORunnerCfg.num_steps_per_env).")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help=(
            "Total parallel envs in the single FrankaDieLiftJointD12D20MixedEnvCfg mixed env (Task 5 - "
            "BACKLOG.md's 2026-07-19 controller decision), split ~evenly between d12/d20 via its own "
            "deterministic round-robin (matches this project's existing PPO training convention's own "
            "num_envs default; NOTE this is a per-shape sample-count reduction vs. Task 5's original "
            "two-envs-side-by-side design, which gave each shape its own full num_envs - see "
            "scripts/distill_specialists.py's own dispatch report for why this tradeoff was accepted)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8192, help="Minibatch size for each BC regression step, drawn from the pooled+shuffled two-shape buffer.")
    parser.add_argument("--num-epochs-per-iteration", type=int, default=4, help="Number of passes over the pooled buffer's minibatches per DAgger iteration.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
        help=(
            "Adam learning rate for the student actor's BC regression (higher than PPO's 1e-4 default: "
            "supervised regression on a fixed/slowly-drifting target tolerates a larger step size than "
            "on-policy PPO)."
        ),
    )
    parser.add_argument("--beta-start", type=float, default=1.0, help="DAgger mixture beta at iteration 0 (1.0 = pure teacher rollout, i.e. a pure-BC warm start).")
    parser.add_argument("--beta-end", type=float, default=0.0, help="DAgger mixture beta at the final iteration (0.0 = pure student rollout).")
    parser.add_argument("--checkpoint-cache-dir", type=str, default=DEFAULT_CHECKPOINT_CACHE_DIR, help="Local cache dir for gs:// teacher checkpoint downloads.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to write the distilled student checkpoint(s) to.")
    parser.add_argument("--save-interval", type=int, default=50, help="Save a student checkpoint every N iterations.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for student init + DAgger mixture sampling.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Mechanical smoke test: real checkpoint download + load + shape-check, real student "
            "ActorCritic construction, but stub synthetic envs stand in for the real Isaac Lab "
            "single-shape envs - no Isaac Sim launch. This task's own deliverable is verified via "
            "this flag, not a real training run (that's Task 5)."
        ),
    )
    parser.add_argument("--dry-run-iterations", type=int, default=3, help="DAgger iterations to run under --dry-run (kept tiny).")
    parser.add_argument("--dry-run-steps", type=int, default=4, help="Rollout steps per stub env per --dry-run iteration (kept tiny).")
    parser.add_argument("--dry-run-num-envs", type=int, default=8, help="Parallel stub envs per shape under --dry-run (kept tiny).")
    app_launcher_cls.add_app_launcher_args(parser)
    return parser


def build_real_mixed_env(num_envs: int, device: str, seed: int):
    """Constructs the real Isaac Lab MIXED (d12+d20) env - Task 5's fix for
    BACKLOG.md's 2026-07-19 BLOCKED entry (a second `ManagerBasedRLEnv`
    cannot be constructed in-process after a first one is built or closed,
    in this Isaac Lab installation - confirmed both simultaneously and via
    sequential reopen). Replaces the original two-single-shape-envs design
    (this module's own former `build_real_env`, which built either
    `FrankaDieLiftJointBigEnvCfg` (d20) or `FrankaDieLiftJointD12BigEnvCfg`
    (d12) one at a time) with ONE `FrankaDieLiftJointD12D20MixedEnvCfg`
    (`tasks/franka/dice_lift_joint_env_cfg.py`) that splits `num_envs`
    between both shapes via a deterministic round-robin
    `MultiAssetSpawnerCfg` - see that class's own docstring and
    BACKLOG.md's controller decision "(b) single mixed-population env" for
    the full rationale. Wrapped exactly like this project's own training
    entry point (scripts/train_franka.py) wraps it for rsl_rl. Every
    isaaclab-dependent import lives inside this function (never at module
    level) so `--dry-run`/`--help` never trigger them - only called from
    `main()`'s non-dry-run branch, i.e. only ever exercised by Task 5's
    real run."""
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20MixedEnvCfg

    env_cfg = FrankaDieLiftJointD12D20MixedEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env = ManagerBasedRLEnv(cfg=env_cfg)
    return RslRlVecEnvWrapper(env, clip_actions=None)


def main(args_cli: argparse.Namespace) -> None:
    torch.manual_seed(args_cli.seed)
    generator = torch.Generator(device="cpu").manual_seed(args_cli.seed)

    if args_cli.dry_run:
        device = "cpu"
        obs_dim_ref, num_actions_ref = REFERENCE_OBS_DIM, REFERENCE_NUM_ACTIONS
        shape_classes = ("d12", "d20")

        print("[dry-run] downloading + shape-checking real teacher checkpoints...")
        d20_obs_dim, d20_num_actions = inspect_checkpoint_shapes(args_cli.d20_checkpoint, args_cli.checkpoint_cache_dir)
        d12_obs_dim, d12_num_actions = inspect_checkpoint_shapes(args_cli.d12_checkpoint, args_cli.checkpoint_cache_dir)
        print(f"[dry-run] d20 checkpoint: obs_dim={d20_obs_dim} num_actions={d20_num_actions}")
        print(f"[dry-run] d12 checkpoint: obs_dim={d12_obs_dim} num_actions={d12_num_actions}")
        if d20_obs_dim != d12_obs_dim or d20_num_actions != d12_num_actions:
            raise ValueError(
                f"teacher checkpoints are NOT shape-compatible: d20 (obs={d20_obs_dim}, act={d20_num_actions}) "
                f"vs d12 (obs={d12_obs_dim}, act={d12_num_actions})"
            )
        if (d20_obs_dim, d20_num_actions) != (obs_dim_ref, num_actions_ref):
            print(
                f"[dry-run] NOTE: real checkpoint shape ({d20_obs_dim}, {d20_num_actions}) differs from this "
                f"module's REFERENCE_OBS_DIM/REFERENCE_NUM_ACTIONS ({obs_dim_ref}, {num_actions_ref}) - using "
                "the real checkpoint's own shape for the rest of this dry run (the reference constants are "
                "documentation-only, see tasks/franka/distillation.py)."
            )
        obs_dim, num_actions = d20_obs_dim, d20_num_actions

        # NOTE: _SyntheticDieEnv (below) is NOT a faithful reproduction of
        # the real 4-shape/41-dim schema (REFERENCE_SHAPE_ONEHOT_START/DIM
        # describe THAT real layout, d20's own class among {d8,d10,d12,d20}
        # at absolute columns [36:40]) - it's a minimal 2-shape stand-in
        # that places its own onehot block at the END of the observation
        # instead, purely for --dry-run's own internal self-consistency (a
        # real bug found and fixed while first running this smoke test,
        # 2026-07-19: using REFERENCE_SHAPE_ONEHOT_START/DIM here read a
        # slice spanning both random-walk state noise AND part of the
        # onehot block, silently misrouting every state to the wrong
        # teacher or matching no teacher at all). shape_onehot_start/dim
        # are therefore derived from the STUB's own actual layout here, not
        # from the reference constants (which describe the real env
        # `build_real_env`/`compute_shape_onehot_offset` use instead).
        shape_onehot_dim = len(shape_classes)
        shape_onehot_start = obs_dim - shape_onehot_dim

        print("[dry-run] loading frozen teachers...")
        d20_teacher = load_frozen_teacher(args_cli.d20_checkpoint, obs_dim, num_actions, device, args_cli.checkpoint_cache_dir)
        d12_teacher = load_frozen_teacher(args_cli.d12_checkpoint, obs_dim, num_actions, device, args_cli.checkpoint_cache_dir)
        router = MultiShapeTeacherRouter(
            {"d20": actor_inference_fn(d20_teacher), "d12": actor_inference_fn(d12_teacher)},
            shape_classes,
            shape_onehot_start,
            shape_onehot_dim,
        )

        print("[dry-run] building student + stub envs (no Isaac Sim launch)...")
        student = build_student_actor_critic(obs_dim, num_actions, device)
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=args_cli.learning_rate)
        envs = {
            "d20": _SyntheticDieEnv(shape_classes.index("d20"), len(shape_classes), args_cli.dry_run_num_envs, obs_dim, device),
            "d12": _SyntheticDieEnv(shape_classes.index("d12"), len(shape_classes), args_cli.dry_run_num_envs, obs_dim, device),
        }

        for it in range(args_cli.dry_run_iterations):
            beta = dagger_beta_schedule(it, args_cli.dry_run_iterations, args_cli.beta_start, args_cli.beta_end)
            mean_loss = run_dagger_iteration(
                envs, student, router, optimizer, args_cli.dry_run_steps, args_cli.batch_size, args_cli.num_epochs_per_iteration, beta, device, generator
            )
            print(f"[dry-run] iteration {it}: beta={beta:.3f} mean_bc_loss={mean_loss:.6f}")

        out_path = os.path.join(args_cli.output_dir, "distill_dryrun_smoke.pt")
        save_student_checkpoint(student, out_path, args_cli.dry_run_iterations, {"dry_run": True})
        print(f"[dry-run] OK - wrote smoke-test checkpoint to {out_path}")
        return

    # Real run (Task 5 only).
    device = args_cli.device
    print("downloading + shape-checking real teacher checkpoints...")
    d20_obs_dim, d20_num_actions = inspect_checkpoint_shapes(args_cli.d20_checkpoint, args_cli.checkpoint_cache_dir)
    d12_obs_dim, d12_num_actions = inspect_checkpoint_shapes(args_cli.d12_checkpoint, args_cli.checkpoint_cache_dir)
    if d20_obs_dim != d12_obs_dim or d20_num_actions != d12_num_actions:
        raise ValueError(
            f"teacher checkpoints are NOT shape-compatible: d20 (obs={d20_obs_dim}, act={d20_num_actions}) "
            f"vs d12 (obs={d12_obs_dim}, act={d12_num_actions})"
        )
    obs_dim, num_actions = d20_obs_dim, d20_num_actions
    # REAL BUG found and fixed while wiring up Task 5's mixed-env driver
    # (2026-07-19, never previously exercised - the original two-envs
    # design crashed before ever reaching router.relabel with real data,
    # see BACKLOG.md's BLOCKED entry): the real env's shape_class
    # observation term is a 4-dim one-hot over the CANONICAL
    # (d8, d10, d12, d20) order (tasks/franka/shape_observations.py's
    # SHAPE_CLASSES - confirmed live, scripts/_diag_d12d20_mixed_env_check.py,
    # 2026-07-19: shape_class term dim is (4,), argmax index 2 for d12 rows,
    # index 3 for d20 rows), NOT a 2-dim ("d12", "d20")-only encoding.
    # MultiShapeTeacherRouter.relabel's own row-routing loop
    # (`for i, shape in enumerate(self._shape_classes): mask = shape_idx ==
    # i`) requires `shape_classes` to be THE SAME order/length as the
    # observation's own onehot columns - passing the 2-element
    # ("d12", "d20") tuple here (copied from --dry-run's own 2-shape STUB
    # env, which deliberately uses a non-faithful 2-dim onehot layout - see
    # that branch's own NOTE comment) against the real env's 4-dim onehot
    # would have silently produced an all-`None`/crashing relabel (argmax
    # values 2/3 never match loop indices 0/1). Fixed by using the full
    # canonical SHAPE_CLASSES tuple for the real run instead - i=2 ("d12")
    # and i=3 ("d20") then correctly match the real onehot's own columns;
    # i=0/1 ("d8"/"d10") simply never match any row (this mixed env never
    # spawns those shapes) and are safely skipped, exactly like any other
    # registered-but-unused shape.
    shape_classes = SHAPE_CLASSES

    d20_teacher = load_frozen_teacher(args_cli.d20_checkpoint, obs_dim, num_actions, device, args_cli.checkpoint_cache_dir)
    d12_teacher = load_frozen_teacher(args_cli.d12_checkpoint, obs_dim, num_actions, device, args_cli.checkpoint_cache_dir)

    # Task 5 architecture fix (BACKLOG.md's 2026-07-19 controller decision
    # "(b) single mixed-population env" - see that entry and
    # build_real_mixed_env's own docstring for the full trace of why the
    # original two-single-shape-envs design is unusable in this Isaac Lab
    # installation). ONE FrankaDieLiftJointD12D20MixedEnvCfg env is built
    # ONCE for the entire run (not per-iteration/per-shape) - its own
    # deterministic round-robin MultiAssetSpawnerCfg already gives every
    # DAgger iteration a single batch mixing both shapes' visited states,
    # so no separate per-shape env open/close/pool step is needed at all.
    env = build_real_mixed_env(args_cli.num_envs, device, args_cli.seed)
    shape_onehot_start, shape_onehot_dim = compute_shape_onehot_offset(env)

    router = MultiShapeTeacherRouter(
        {"d20": actor_inference_fn(d20_teacher), "d12": actor_inference_fn(d12_teacher)},
        shape_classes,
        shape_onehot_start,
        shape_onehot_dim,
    )
    student = build_student_actor_critic(obs_dim, num_actions, device)
    optimizer = torch.optim.Adam(student.actor.parameters(), lr=args_cli.learning_rate)
    student_action_fn = actor_inference_fn(student)

    try:
        for it in range(args_cli.num_iterations):
            beta = dagger_beta_schedule(it, args_cli.num_iterations, args_cli.beta_start, args_cli.beta_end)

            def _mixture_action_fn(obs: torch.Tensor) -> torch.Tensor:
                student_actions = student_action_fn(obs)
                teacher_actions = router.relabel(obs)
                return mix_actions(student_actions, teacher_actions, beta, generator=generator)

            # A single mixed-population rollout batch (both shapes already
            # present, per-row-routed by MultiShapeTeacherRouter.relabel off
            # each row's own live shape-onehot observation feature) - passed
            # as a one-element list since regress_on_pooled_batches expects a
            # list of per-source batches to pool+shuffle (trivially a no-op
            # concat of one tensor here, still shuffled before regression).
            batch = collect_rollout(env, _mixture_action_fn, args_cli.rollout_steps, device=device)
            mean_loss = regress_on_pooled_batches(
                [batch], student, router, optimizer, args_cli.batch_size, args_cli.num_epochs_per_iteration, generator
            )
            print(f"iteration {it}/{args_cli.num_iterations}: beta={beta:.3f} mean_bc_loss={mean_loss:.6f}")
            if (it + 1) % args_cli.save_interval == 0 or it == args_cli.num_iterations - 1:
                out_path = os.path.join(args_cli.output_dir, f"model_{it}.pt")
                save_student_checkpoint(student, out_path, it)
                print(f"saved checkpoint: {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    parser = build_arg_parser(AppLauncher)
    args_cli = parser.parse_args()

    simulation_app = None
    if not args_cli.dry_run:
        # Real run only (Task 5): actually boot Isaac Sim. --dry-run never
        # reaches this branch, so `--dry-run`/`--help` need no Isaac Sim
        # launch at all (this task's own scope constraint).
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    main(args_cli)

    if simulation_app is not None:
        simulation_app.close()
