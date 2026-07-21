"""Worked demonstration: train PPO on `ArmReachEnv` under multiple action modes.

**Scope note:** part of `toy_env/`, a CPU-only, physics-free proxy environment
— see `toy_env/arm_reach_env.py`'s module docstring and
`kb/wiki/concepts/toy-kinematic-proxy-env.md` for what this is and isn't a
substitute for (short version: fast hypothesis generator for algorithm/
action-space questions, NOT a substitute for re-verifying anything promising
in the real Isaac Lab simulator).

This script is the actual validation that the toy environment is informative,
not just a nice-looking simulator: it trains a real Stable-Baselines3 PPO
agent under each of two or more `action_mode`s (default: `absolute` and
`relative`), and periodically evaluates the in-progress policy (not just the
final one) so a full training-time curve can be inspected for the specific
qualitative pattern this project's real Isaac Lab experiment found — under
**absolute joint-position control**, the policy transiently discovers how to
approach the object, then abandons that behavior over training (a rise-then-
decay shape in the `reaching_object`-style signal); under **relative/delta**
(or task-space) control, that regression either doesn't happen or is much
less severe.

Usage
-----
    toy_env/.venv/bin/python -m toy_env.train_demo \\
        --modes absolute relative --timesteps 150000 --seed 0 \\
        --out-dir toy_env/runs

Run from the repo root (so `toy_env` is importable as a package) with the
project's own dedicated venv (`toy_env/.venv`), NOT system python and NOT
`vision/.venv` — neither has the packages this script needs, and this proxy
environment is deliberately its own runtime, isolated from both of this
repo's other Python environments (see this module's own kb article for why).

What gets recorded, per action mode
------------------------------------
`EvalRecorderCallback` runs `n_eval_episodes` fixed-seed evaluation rollouts
of the *current* policy every `eval_every` training timesteps (deterministic
actions, no exploration noise) and records, at each checkpoint:

- `mean_min_dist`: mean, across eval episodes, of the closest the
  end-effector ever got to the target during that episode. This is this toy
  environment's analogue of the real experiment's `reaching_object` reward
  curve — a policy that "discovers then abandons" reaching should show this
  curve improve (decrease) early, then get worse (increase) later.
  `success_rate`: fraction of eval episodes reaching a sustained lift.

This is deliberately a *separate*, deterministic, fixed-seed evaluation
loop — not read off the training rollout's own (stochastic, exploring)
reward — so the recorded curve reflects the policy's own actual competence
at each checkpoint, not exploration noise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from toy_env.arm_reach_env import ArmReachEnv


class EvalRecorderCallback(BaseCallback):
    """Periodic fixed-seed deterministic eval, recording a training-time curve.

    See module docstring for why this exists and what `mean_min_dist` means.
    """

    def __init__(self, action_mode: str, eval_every: int, n_eval_episodes: int = 8, verbose: int = 0):
        super().__init__(verbose)
        self.action_mode = action_mode
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.history: list[dict] = []
        self._last_eval = 0

    def _on_training_start(self) -> None:
        self._last_eval = 0
        self._run_eval()  # record the untrained baseline too

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_every:
            self._last_eval = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self) -> None:
        env = ArmReachEnv(action_mode=self.action_mode)
        min_dists = []
        successes = []
        for ep in range(self.n_eval_episodes):
            obs, info = env.reset(seed=10_000 + ep)
            min_dist = info["dist"]
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                min_dist = min(min_dist, info["dist"])
            min_dists.append(min_dist)
            successes.append(bool(info["success"]))
        record = {
            "timestep": int(self.num_timesteps),
            "mean_min_dist": float(np.mean(min_dists)),
            "success_rate": float(np.mean(successes)),
        }
        self.history.append(record)
        if self.verbose:
            print(
                f"[eval @ {record['timestep']:>7d}] "
                f"mean_min_dist={record['mean_min_dist']:.4f}  "
                f"success_rate={record['success_rate']:.3f}"
            )


def train_one(
    action_mode: str,
    algo: str,
    total_timesteps: int,
    seed: int,
    out_dir: Path,
    n_envs: int = 4,
) -> dict:
    """Train one PPO (or SAC) run for a given action mode; save model + history.

    Hyperparameters (`n_steps=128`, `n_epochs=5`, `n_envs=4` default) are
    deliberately modest, tuned for this environment's own tiny CPU cost on a
    resource-constrained host (Raspberry Pi, 4 cores, ~1GB free RAM) rather
    than tuned for best possible sample efficiency — this is a fast proxy
    demo, not a hyperparameter-optimized recipe. `SB3`'s `DummyVecEnv` (used
    by `make_vec_env`'s default) steps all `n_envs` sequentially in a single
    process, so wall-clock cost scales close to linearly with `n_envs`, not
    for free the way true multi-core parallelism would be.
    """

    def _make() -> ArmReachEnv:
        return Monitor(ArmReachEnv(action_mode=action_mode))

    vec_env = make_vec_env(_make, n_envs=n_envs, seed=seed)

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=seed,
            n_steps=128,
            batch_size=128,
            n_epochs=5,
            learning_rate=3e-4,
        )
    elif algo == "sac":
        from stable_baselines3 import SAC

        model = SAC("MlpPolicy", vec_env, verbose=0, seed=seed)
    else:
        raise ValueError(f"unsupported algo {algo!r}")

    # Fixed eval cadence (not scaled with total_timesteps) - eval overhead
    # (n_eval_episodes full rollouts, deterministic, single env) is
    # per-checkpoint-constant, so scaling checkpoint *count* with
    # total_timesteps would make eval overhead dominate total wall time for
    # longer runs. ~12-15 checkpoints is enough to see a training-time curve
    # shape without that blowup.
    eval_every = max(1500, total_timesteps // 15)
    cb = EvalRecorderCallback(action_mode, eval_every=eval_every, n_eval_episodes=5, verbose=1)

    print(f"=== training action_mode={action_mode!r} algo={algo!r} timesteps={total_timesteps} ===")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    elapsed = time.time() - t0
    print(f"=== done: action_mode={action_mode!r} in {elapsed:.1f}s ===")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{action_mode}_{algo}_seed{seed}.zip"
    model.save(str(model_path))

    result = {
        "action_mode": action_mode,
        "algo": algo,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "elapsed_sec": elapsed,
        "n_envs": n_envs,
        "history": cb.history,
        "model_path": str(model_path),
    }
    history_path = out_dir / f"{action_mode}_{algo}_seed{seed}_history.json"
    history_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {history_path}")
    return result


def summarize(results: list[dict]) -> None:
    """Print a plain-text verdict: does 'absolute' show rise-then-decay that
    'relative' (or task_space) doesn't?"""
    print("\n=== Summary: mean_min_dist(t) per action mode ===")
    for r in results:
        hist = r["history"]
        dists = [h["mean_min_dist"] for h in hist]
        best_idx = int(np.argmin(dists))
        best_t = hist[best_idx]["timestep"]
        final = dists[-1]
        best = dists[best_idx]
        regression = final - best
        pct_regression = (regression / best * 100.0) if best > 1e-9 else float("nan")
        print(
            f"  {r['action_mode']:>10s}: best={best:.4f} @ t={best_t} "
            f"(of {r['total_timesteps']}), final={final:.4f}, "
            f"regression(final-best)={regression:+.4f} ({pct_regression:+.1f}% of best), "
            f"final success_rate={hist[-1]['success_rate']:.3f}"
        )
    print(
        "\nInterpretation guide: a large positive 'regression' with best "
        "reached well before the final timestep is the toy-scale analogue of "
        "the real Isaac Lab 'transient discovery then abandonment' pattern. "
        "A regression near zero (best ~= final, or best reached late/at the "
        "end) means that mode did NOT show the pathology in this run."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["absolute", "relative"], choices=["absolute", "relative", "task_space"])
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--out-dir", default="toy_env/runs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = []
    for mode in args.modes:
        results.append(
            train_one(mode, args.algo, args.timesteps, args.seed, out_dir, n_envs=args.n_envs)
        )

    summarize(results)

    combined_path = out_dir / f"comparison_{args.algo}_seed{args.seed}.json"
    combined_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote combined comparison to {combined_path}")


if __name__ == "__main__":
    main()
