"""Config-override resolver shared by scripts/train.py and the sweep
framework. Pure python (no isaaclab import) so it is unit-testable on its
own.

An override is a flat dict of dotted-path keys -> JSON scalar values. Each
key starts with a namespace ('env.' or 'agent.'); the remainder is a dotted
path under that cfg object. Each path segment resolves as a dict key when
the current object is a dict, otherwise as an attribute - so both
configclass fields (attributes) and reward-term ``params`` dict entries
(e.g. '...params.touch_std') are reachable with the same syntax.
"""

from __future__ import annotations


def set_dotted(root, dotted_path: str, value) -> None:
    segments = dotted_path.split(".")
    cur = root
    for seg in segments[:-1]:
        cur = cur[seg] if isinstance(cur, dict) else getattr(cur, seg)
    last = segments[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def apply_overrides(env_cfg, agent_cfg, overrides: dict, verbose: bool = True) -> None:
    """Apply a flat dict of dotted-path overrides to the constructed cfgs.
    Fails loud on an unknown namespace or empty path - a silently-ignored
    override would invalidate a trial's recorded parameter vector."""
    for key, value in overrides.items():
        namespace, _, rest = key.partition(".")
        if not rest:
            raise ValueError(f"Override key {key!r} has no field path after the namespace.")
        if namespace == "env":
            set_dotted(env_cfg, rest, value)
        elif namespace == "agent":
            set_dotted(agent_cfg, rest, value)
        else:
            raise ValueError(f"Override key {key!r} must start with 'env.' or 'agent.', got {namespace!r}.")
        if verbose:
            print(f"[override] {key} = {value!r}")
