# toy_env

A CPU-only, physics-free proxy environment for fast RL prototyping — no
Isaac Sim, no GPU, no cloud cost. Built to let algorithm/exploration/
action-space questions be iterated in seconds-to-minutes before committing
real Isaac Sim GPU time to anything promising.

**Full design rationale, scope, and limits:** see the module docstrings in
`kinematic_arm.py` and `arm_reach_env.py`, and
`kb/wiki/concepts/toy-kinematic-proxy-env.md`. Read those before trusting
any result from this environment for a real decision — short version: this
is a hypothesis generator, not a conclusion; it has no contact physics, no
friction, and no real grasp mechanics, and anything interesting found here
still needs re-verification in the real Isaac Lab simulator.

## Setup (own dedicated venv — do not reuse `vision/.venv` or `isaaclab.sh`)

```bash
python3 -m venv toy_env/.venv
# CPU-only torch index first — plain `pip install torch` on this repo's
# aarch64 Raspberry Pi host pulls in ~1.5GB of unused NVIDIA cuda-toolkit/
# cudnn packages; the Pi has no NVIDIA GPU at all.
toy_env/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
toy_env/.venv/bin/pip install -r toy_env/requirements.txt
```

## Run the tests

```bash
toy_env/.venv/bin/pytest toy_env/tests/ -v
```

## Train the worked demonstration (PPO across action modes)

```bash
toy_env/.venv/bin/python -m toy_env.train_demo \
    --modes absolute relative task_space --algo ppo \
    --timesteps 100000 --seed 0 --out-dir toy_env/runs
```

Trains one PPO run per action mode, recording a training-time curve of
"closest approach to target" at fixed-seed evaluation checkpoints — the
toy-scale analogue of the real Isaac Lab `reaching_object` reward curve —
so a rise-then-decay pathology (or its absence) can be read off directly.
Results land in `toy_env/runs/` (gitignored — model checkpoints + per-mode
JSON history + a combined comparison JSON).

## Visualize a rollout

```python
from toy_env.arm_reach_env import ArmReachEnv
from toy_env.visualize import plot_arm_pose_3d, render_episode_gif

env = ArmReachEnv(action_mode="task_space", seed=0)
render_episode_gif(env, out_path="toy_env/renders/demo.gif")  # random policy
```

Pass a trained SB3 model's `.predict` as `policy=model.predict` to visualize
a trained rollout instead of a random one. `toy_env/renders/` is gitignored.

## Files

- `kinematic_arm.py` — pure forward-kinematics 7-joint arm chain + a
  numerical Jacobian (no dynamics, no collision).
- `arm_reach_env.py` — the Gymnasium env: 3 action modes (`absolute`,
  `relative`, `task_space`), staged reach→grip→lift reward.
- `visualize.py` — matplotlib 3D static plots + GIF rollout rendering.
- `train_demo.py` — the worked PPO demonstration + cross-mode comparison.
- `tests/` — pytest sanity/regression tests for the kinematics and env.
