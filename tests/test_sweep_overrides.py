"""Sim-independent unit tests for sweeps/overrides.py's config-override
resolver (set_dotted / apply_overrides) - pure python, no isaaclab import
needed, run with plain pytest:
`pytest tests/test_sweep_overrides.py -v`
"""

import pytest

from sweeps.overrides import apply_overrides, set_dotted


class _Inner:
    def __init__(self):
        self.weight = 1.0
        self.params = {"touch_std": 0.05, "nested": {"deep": 1}}


class _Cfg:
    def __init__(self):
        self.rewards = {"touch_goal_milestone_bonus": _Inner()}
        self.actions = _Actions()
        self.episode_length_s = 20.0


class _Actions:
    def __init__(self):
        self.joint_positions = _JointPositions()


class _JointPositions:
    def __init__(self):
        self.scale = 0.5


# ---------------------------------------------------------------------------
# set_dotted
# ---------------------------------------------------------------------------


def test_set_dotted_nested_attribute_path():
    """A pure attribute chain (env.actions.joint_positions.scale) should set
    the leaf attribute, not touch anything else."""
    cfg = _Cfg()
    set_dotted(cfg, "actions.joint_positions.scale", 0.75)
    assert cfg.actions.joint_positions.scale == 0.75


def test_set_dotted_dict_key_path():
    """A path that passes through a dict (rewards.<term>.params dict-key
    entries) must resolve dict keys where the current object is a dict and
    attributes otherwise, mixed in the same path."""
    cfg = _Cfg()
    set_dotted(cfg, "rewards.touch_goal_milestone_bonus.params.touch_std", 0.09)
    assert cfg.rewards["touch_goal_milestone_bonus"].params["touch_std"] == 0.09
    # sibling values untouched
    assert cfg.rewards["touch_goal_milestone_bonus"].weight == 1.0


def test_set_dotted_dict_key_then_attribute():
    """Path resolves a dict key (rewards -> term), then an attribute (.weight)
    on the object found there."""
    cfg = _Cfg()
    set_dotted(cfg, "rewards.touch_goal_milestone_bonus.weight", 42.0)
    assert cfg.rewards["touch_goal_milestone_bonus"].weight == 42.0


def test_set_dotted_nested_dict_within_dict():
    """A deeper dict-within-dict path (params.nested.deep) resolves through
    two consecutive dict-key segments."""
    cfg = _Cfg()
    set_dotted(cfg, "rewards.touch_goal_milestone_bonus.params.nested.deep", 99)
    assert cfg.rewards["touch_goal_milestone_bonus"].params["nested"]["deep"] == 99


def test_set_dotted_single_segment_top_level_attribute():
    """A one-segment path is a direct attribute set on root with no
    traversal."""
    cfg = _Cfg()
    set_dotted(cfg, "episode_length_s", 35.0)
    assert cfg.episode_length_s == 35.0


# ---------------------------------------------------------------------------
# apply_overrides - namespace split
# ---------------------------------------------------------------------------


def test_apply_overrides_splits_env_and_agent_namespaces():
    env_cfg = _Cfg()
    agent_cfg = _Cfg()
    overrides = {
        "env.actions.joint_positions.scale": 0.8,
        "agent.actions.joint_positions.scale": 0.3,
    }
    apply_overrides(env_cfg, agent_cfg, overrides, verbose=False)
    assert env_cfg.actions.joint_positions.scale == 0.8
    assert agent_cfg.actions.joint_positions.scale == 0.3


def test_apply_overrides_dict_path_under_env_namespace():
    env_cfg = _Cfg()
    agent_cfg = _Cfg()
    overrides = {"env.rewards.touch_goal_milestone_bonus.params.touch_std": 0.11}
    apply_overrides(env_cfg, agent_cfg, overrides, verbose=False)
    assert env_cfg.rewards["touch_goal_milestone_bonus"].params["touch_std"] == 0.11
    # agent_cfg must be untouched
    assert agent_cfg.rewards["touch_goal_milestone_bonus"].params["touch_std"] == 0.05


def test_apply_overrides_multiple_keys_applied_independently():
    env_cfg = _Cfg()
    agent_cfg = _Cfg()
    overrides = {
        "env.episode_length_s": 40.0,
        "env.rewards.touch_goal_milestone_bonus.weight": 12.0,
    }
    apply_overrides(env_cfg, agent_cfg, overrides, verbose=False)
    assert env_cfg.episode_length_s == 40.0
    assert env_cfg.rewards["touch_goal_milestone_bonus"].weight == 12.0


# ---------------------------------------------------------------------------
# apply_overrides - fail-loud behavior (deliberate per the module docstring)
# ---------------------------------------------------------------------------


def test_apply_overrides_unknown_namespace_raises():
    env_cfg = _Cfg()
    agent_cfg = _Cfg()
    with pytest.raises(ValueError, match="must start with 'env.' or 'agent.'"):
        apply_overrides(env_cfg, agent_cfg, {"model.foo": 1.0}, verbose=False)


def test_apply_overrides_empty_path_after_namespace_raises():
    """A key with no dot at all (e.g. 'env') partitions to namespace='env',
    rest='' - must raise rather than silently no-op, since a silently
    ignored override would invalidate a trial's recorded parameter vector."""
    env_cfg = _Cfg()
    agent_cfg = _Cfg()
    with pytest.raises(ValueError, match="no field path after the namespace"):
        apply_overrides(env_cfg, agent_cfg, {"env": 1.0}, verbose=False)


def test_apply_overrides_bare_namespace_with_trailing_dot_raises():
    """'env.' (trailing dot, nothing after) must also raise - same
    empty-rest fail-loud path as the no-dot case."""
    env_cfg = _Cfg()
    agent_cfg = _Cfg()
    with pytest.raises(ValueError, match="no field path after the namespace"):
        apply_overrides(env_cfg, agent_cfg, {"env.": 1.0}, verbose=False)
