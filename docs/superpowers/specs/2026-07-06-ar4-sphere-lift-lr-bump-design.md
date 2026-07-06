# AR4 sphere lift: SA-PPO-style dynamic learning-rate bump

## Problem

Three real attempts this session (sparse-only, curriculum-gated dense,
always-on dense) all produce the identical outcome: the policy reliably
reaches and grips the sphere (`grasp_contact` converges to ~92% per-step
by iteration ~600-750) and then holds a completely static pose for the
rest of every episode — the sphere never leaves the ground.
`lifting_sphere` stays at exactly `0.0000` (or noise-level) in every run,
regardless of whether a dense shaping term is present, curriculum-gated,
or always-on.

Citation-verified literature research
(`docs/superpowers/specs/research/2026-07-06-lift-reward-literature-junior.md`,
`-senior-review.md`) identifies the likely mechanism as **PPO entropy
collapse**: once a safe, reward-sufficient behavior is found, policy
entropy drops and exploration of riskier alternatives (like lifting)
effectively stops — a dense reward term nudging toward a further goal
doesn't reintroduce exploration on its own if the policy has already
stopped exploring. The strongest verified citation found is
**Li et al., *Sensors* 2025, 25(17):5253 (DOI 10.3390/s25175253)**, a
robotic-arm-grasping PPO paper explicitly targeting "local optimum traps"
via a simulated-annealing+PPO hybrid (SA-PPO) with a **dynamically-
adjusted learning rate**, reporting a 92%→98% success-rate improvement
over baseline PPO with real-robot validation — the single most directly
applicable finding this session has produced.

Per user instruction, this experiment (and a second, independent one on
potential-based reward shaping) will both be tried, sequentially and as
single-variable tests — not combined, so it stays clear which one (if
either) actually works.

## A real, load-bearing detail found while designing this

This repo's current PPO config (`tasks/ar4/agents/rsl_rl_ppo_cfg.py`) uses
`schedule="adaptive"` with `desired_kl=0.01` — `rsl_rl`'s adaptive
schedule adjusts the learning rate based on the KL divergence between the
old and new policy each update, **increasing** LR when actual KL exceeds
`desired_kl` and **decreasing** it otherwise. Once a policy has converged
(as ours has, on the static-grip behavior), KL divergence between
successive updates is naturally low — meaning the *existing* adaptive
schedule is already actively driving the learning rate down over time as
the policy stabilizes, the opposite of what's needed to escape a local
optimum. Simply setting a higher starting `learning_rate` while leaving
`schedule="adaptive"` would very likely get corrected back down within a
few updates once the (already fairly stable) post-convergence policy
shows low KL divergence again, defeating the intervention. **The
schedule itself must be switched to `"fixed"` for this experiment**, not
just the learning rate value bumped, or the fix has no chance to persist
long enough to matter.

## Design

### 1. Reuse an existing checkpoint instead of re-running phase 1

`logs/train/2026-07-06_12-24-08/` (the just-completed always-on-lift
run) already has `model_700.pt` — exactly the "grip converged, exploration
about to collapse" state this experiment needs to intervene at
(`grasp_contact` was already at ~17.8/20 by iteration 700 in every run
this session). Reusing it:
- Saves re-running an entire ~16-minute training phase for no new
  information (phase 1's behavior at this reward config is already
  characterized in detail).
- Isolates the learning-rate bump as the **only** new variable —  the
  reward function stays exactly as it already is (the always-on
  `lift_height_progress` term, weight 25.0, already active and already
  characterized as producing a negligible effect on its own). This
  experiment tests purely "does a late learning-rate bump let the policy
  escape, holding the reward function fixed," not a combination of reward
  and LR changes.

### 2. New script: `scripts/train_lr_bump.py`

Mirrors `scripts/train.py`'s structure (same `AppLauncher` boilerplate,
same env/agent config classes) but adds checkpoint-resume + hyperparameter
override:

```python
"""Continue AR4 sphere pick-and-place PPO training from an existing
checkpoint with a bumped, fixed learning rate - an SA-PPO-style
intervention (Li et al., Sensors 2025, DOI 10.3390/s25175253) to escape
the "reach, grip, freeze" local optimum every prior experiment this
session has hit. See
docs/superpowers/specs/2026-07-06-ar4-sphere-lift-lr-bump-design.md.

Does NOT change the reward function - reuses whatever checkpoint is
passed in as-is, testing the learning-rate bump in isolation.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_lr_bump.py \
        --checkpoint logs/train/2026-07-06_12-24-08/model_700.pt \
        --num_envs 4096 --max_iterations 1500 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Continue AR4 pick-and-place PPO training with a bumped learning rate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to resume from.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=1500, help="Additional learning iterations beyond the checkpoint's own iteration count.")
parser.add_argument("--learning_rate", type=float, default=1.0e-3, help="Bumped, fixed learning rate for this phase.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "train")


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device
    # The bump: fixed (not adaptive) schedule so it isn't corrected back
    # down by KL-divergence feedback from the already-converged policy.
    agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    agent_cfg.algorithm.schedule = "fixed"
    env_cfg.seed = agent_cfg.seed

    log_dir = os.path.join(LOG_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    print(f"Resumed from {args_cli.checkpoint} at iteration {runner.current_learning_iteration}")

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"Training complete. Checkpoints and logs written to: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

Key points verified directly against the installed `rsl_rl` source
(`OnPolicyRunner.load()`/`.learn()`, not assumed):
- `runner.load(path)` restores model weights, optimizer state, **and**
  `current_learning_iteration` from the checkpoint's saved `iter` field.
- `learn(num_learning_iterations=N)` computes `tot_iter = start_iter + N`
  — i.e. `N` is **additional** iterations on top of the loaded
  checkpoint's iteration count, not a total. Passing `--max_iterations
  1500` after resuming from iteration 700 runs through iteration 2200,
  matching a full fresh run's worth of *additional* training time given
  to the bumped-LR phase.
- A fresh `log_dir` (new timestamp) is used so this phase's TensorBoard
  event file doesn't collide with the original run's; the new event file
  logs scalars from step 700 onward (not from 0), which is expected and
  fine.

### 3. Not touched

The reward function (`tasks/ar4/mdp.py`), `RewardsCfg`,
`agent_cfg.algorithm.entropy_coef` (stays `0.006`, unchanged — only
`learning_rate`/`schedule` change), and every other hyperparameter stay
exactly as they are. `scripts/train.py` itself is not modified — this is
a new, separate script, since the checkpoint-resume + hyperparameter-
override flow is a genuinely different mode of operation, not a flag
worth adding to the main training entry point for a one-off experiment.

## Verification plan

Smoke test first with a short resumed run
(`--max_iterations 2 --num_envs 16`) to confirm the resume/override
mechanics work before committing to the full 1500-iteration continuation.
Then the full run (`--num_envs 4096 --max_iterations 1500`), monitoring
`Episode_Reward/lifting_sphere`, `Episode_Reward/grasp_contact`,
`Episode_Reward/lift_height_progress`, `Loss/learning_rate` (to directly
confirm the bump held near `1e-3` and didn't get corrected back down -
this is the one thing that could silently invalidate the whole
experiment, so it's checked explicitly, not assumed), and
`Episode_Termination/sphere_reached_goal` via TensorBoard. Then real eval
(`--episodes 10`) with frame-extracted video inspection of all 10
episodes, same rigor as every prior experiment.

If `lifting_sphere` still doesn't move and/or the eval video still shows
the same "reach, grip, freeze" signature, this is the fourth real attempt
on this sub-problem — proceed directly to the second planned experiment
(potential-based reward shaping) regardless, per the user's "try both"
instruction, rather than pausing to ask.
