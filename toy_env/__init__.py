"""toy_env: a CPU-only, physics-free proxy environment for fast RL prototyping.

See the package-level docstrings in `kinematic_arm.py` and `arm_reach_env.py`,
and `kb/wiki/concepts/toy-kinematic-proxy-env.md`, for what this is and is
not a substitute for. In short: no Isaac Sim, no GPU, no contact physics —
a cheap hypothesis-generator for algorithm/action-space questions, not a
conclusion-generator. Any promising finding here still needs re-verification
in the real Isaac Lab simulator.
"""
