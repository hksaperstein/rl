"""Sim-independent unit tests for sweeps/strategies.py's HillclimbStrategy,
RandomStrategy, and GridStrategy - pure python, no isaaclab import needed,
run with plain pytest: `pytest tests/test_sweep_strategies.py -v`
"""

from sweeps.runner import TrialResult
from sweeps.spaces import ParameterSpec, TaskSpace
from sweeps.strategies import GridStrategy, HillclimbStrategy, RandomStrategy

_SPEC_WEIGHT = ParameterSpec(
    name="weight",
    keys=("env.rewards.foo.weight",),
    baseline=10.0,
    bounds=(5.0, 20.0),
    step=1.5,
    step_mode="mult",
)
_SPEC_STD = ParameterSpec(
    name="std",
    keys=("env.rewards.foo.params.std",),
    baseline=0.05,
    bounds=(0.02, 0.15),
    step=0.02,
    step_mode="add",
)
_SPEC_LR = ParameterSpec(
    name="lr",
    keys=("agent.algorithm.learning_rate",),
    baseline=1.0e-4,
    bounds=(3.0e-5, 3.0e-4),
    step=2.0,
    step_mode="mult",
    scale="log",
)

_SPACE = TaskSpace(
    name="test",
    train_flag="--test",
    success_metric="Episode_Termination/x",
    stability_metric="Loss/value_function",
    parameters=(_SPEC_WEIGHT, _SPEC_STD, _SPEC_LR),
)


def _result(success=0.5, stability=0.0, completed=True, traceback=False, unstable=False, run_dir="/tmp/x"):
    return TrialResult(
        success_metric=success,
        stability_metric=stability,
        run_dir=run_dir,
        returncode=0,
        completed=completed,
        traceback_found=traceback,
        unstable=unstable,
    )


# ---------------------------------------------------------------------------
# HillclimbStrategy
# ---------------------------------------------------------------------------


def test_hillclimb_round_zero_is_baseline():
    strat = HillclimbStrategy(_SPACE, rounds=3, seed=0)
    pv = strat.next_trial()
    assert pv == {}


def test_hillclimb_exhausts_after_declared_rounds():
    strat = HillclimbStrategy(_SPACE, rounds=2, seed=0)
    assert strat.next_trial() is not None  # round 0 (baseline)
    assert strat.next_trial() is not None  # round 1
    assert strat.next_trial() is None  # exhausted


def test_hillclimb_proposes_single_param_within_bounds():
    strat = HillclimbStrategy(_SPACE, rounds=3, param_names=["weight"], seed=0)
    strat.next_trial()  # baseline
    pv = strat.next_trial()
    assert list(pv.keys()) == ["weight"]
    lo, hi = _SPEC_WEIGHT.bounds
    assert lo <= pv["weight"] <= hi
    # a real step should have been taken (not stuck at baseline)
    assert pv["weight"] != _SPEC_WEIGHT.baseline


def test_hillclimb_record_baseline_sets_best_success():
    strat = HillclimbStrategy(_SPACE, rounds=3, seed=0)
    pv0 = strat.next_trial()
    outcome = strat.record(pv0, _result(success=0.5))
    assert outcome == "BASELINE"
    assert strat.best_success == 0.5
    assert strat.best_values == {}


def test_hillclimb_adopts_on_improvement():
    strat = HillclimbStrategy(_SPACE, rounds=3, param_names=["weight"], seed=0)
    pv0 = strat.next_trial()
    strat.record(pv0, _result(success=0.5))
    pv1 = strat.next_trial()
    name, value = next(iter(pv1.items()))
    outcome = strat.record(pv1, _result(success=0.9))
    assert outcome == "KEPT"
    assert strat.best_values[name] == value
    assert strat.best_success == 0.9


def test_hillclimb_rejects_on_regression():
    strat = HillclimbStrategy(_SPACE, rounds=3, param_names=["weight"], seed=0)
    pv0 = strat.next_trial()
    strat.record(pv0, _result(success=0.5))
    pv1 = strat.next_trial()
    outcome = strat.record(pv1, _result(success=0.3))  # worse than 0.5
    assert outcome == "REVERTED"
    assert strat.best_values == {}  # not adopted
    assert strat.best_success == 0.5  # unchanged


def test_hillclimb_rejects_on_instability_even_if_success_looks_better():
    strat = HillclimbStrategy(_SPACE, rounds=3, param_names=["weight"], seed=0)
    pv0 = strat.next_trial()
    strat.record(pv0, _result(success=0.5))
    pv1 = strat.next_trial()
    # success_metric looks like an improvement, but unstable must veto adoption
    outcome = strat.record(pv1, _result(success=0.99, unstable=True))
    assert outcome == "UNSTABLE"
    assert strat.best_values == {}
    assert strat.best_success == 0.5


def test_hillclimb_rejects_on_error_even_if_success_looks_better():
    strat = HillclimbStrategy(_SPACE, rounds=3, param_names=["weight"], seed=0)
    pv0 = strat.next_trial()
    strat.record(pv0, _result(success=0.5))
    pv1 = strat.next_trial()
    outcome = strat.record(pv1, _result(success=0.99, completed=False))
    assert outcome == "ERROR"
    assert strat.best_values == {}
    assert strat.best_success == 0.5


def test_hillclimb_round_robins_across_declared_params():
    strat = HillclimbStrategy(_SPACE, rounds=5, param_names=["weight", "std"], seed=0)
    strat.next_trial()  # baseline
    pv1 = strat.next_trial()
    pv2 = strat.next_trial()
    assert list(pv1.keys()) == ["weight"]
    assert list(pv2.keys()) == ["std"]


# ---------------------------------------------------------------------------
# RandomStrategy
# ---------------------------------------------------------------------------


def test_random_strategy_linear_bounds_respected():
    strat = RandomStrategy(_SPACE, n_trials=50, param_names=["weight"], seed=1)
    values = []
    while (pv := strat.next_trial()) is not None:
        values.append(pv["weight"])
    assert len(values) == 50
    lo, hi = _SPEC_WEIGHT.bounds
    assert all(lo <= v <= hi for v in values)
    assert len(set(values)) > 1  # genuine randomness, not a constant


def test_random_strategy_log_scale_bounds_respected():
    strat = RandomStrategy(_SPACE, n_trials=200, param_names=["lr"], seed=2)
    values = []
    while (pv := strat.next_trial()) is not None:
        values.append(pv["lr"])
    lo, hi = _SPEC_LR.bounds
    assert all(lo <= v <= hi for v in values)
    # bounds span a full order of magnitude (3e-5..3e-4); log sampling
    # across 200 draws should produce values spanning a meaningful range,
    # unlike linear sampling which would cluster near the upper end.
    assert max(values) / min(values) > 3


def test_random_strategy_exhausts_after_n_trials():
    strat = RandomStrategy(_SPACE, n_trials=3, seed=0)
    for _ in range(3):
        assert strat.next_trial() is not None
    assert strat.next_trial() is None


def test_random_strategy_covers_all_declared_param_names_by_default():
    strat = RandomStrategy(_SPACE, n_trials=1, seed=0)
    pv = strat.next_trial()
    assert set(pv.keys()) == {p.name for p in _SPACE.parameters}


def test_random_strategy_record_labels():
    strat = RandomStrategy(_SPACE, n_trials=1, seed=0)
    assert strat.record({}, _result(completed=False)) == "ERROR"
    assert strat.record({}, _result(unstable=True)) == "UNSTABLE"
    assert strat.record({}, _result()) == "SAMPLED"


# ---------------------------------------------------------------------------
# GridStrategy
# ---------------------------------------------------------------------------


def test_grid_strategy_cartesian_enumeration_within_bounds():
    strat = GridStrategy(_SPACE, param_names=["weight", "std"], points_per_param=3, max_trials=32)
    combos = []
    while (pv := strat.next_trial()) is not None:
        combos.append(pv)
    assert len(combos) == strat.total()
    assert strat.total() <= 9  # 3x3 cartesian product cap
    assert all(set(c.keys()) == {"weight", "std"} for c in combos)
    wlo, whi = _SPEC_WEIGHT.bounds
    slo, shi = _SPEC_STD.bounds
    assert all(wlo <= c["weight"] <= whi for c in combos)
    assert all(slo <= c["std"] <= shi for c in combos)
    # every combo distinct
    assert len(combos) == len({tuple(sorted(c.items())) for c in combos})


def test_grid_strategy_max_trials_caps_full_cartesian_product():
    strat = GridStrategy(_SPACE, param_names=["weight", "std"], points_per_param=5, max_trials=4)
    assert strat.total() == 4  # 5x5=25 combos truncated to 4
    count = 0
    while strat.next_trial() is not None:
        count += 1
    assert count == 4


def test_grid_strategy_single_point_for_degenerate_resolution():
    strat = GridStrategy(_SPACE, param_names=["weight"], points_per_param=1, max_trials=32)
    assert strat.total() == 1
    pv = strat.next_trial()
    lo, _hi = _SPEC_WEIGHT.bounds
    assert pv["weight"] == lo


def test_grid_strategy_exhausts_and_record_labels():
    strat = GridStrategy(_SPACE, param_names=["weight"], points_per_param=2, max_trials=32)
    while strat.next_trial() is not None:
        pass
    assert strat.next_trial() is None
    assert strat.record({}, _result(completed=False)) == "ERROR"
    assert strat.record({}, _result(unstable=True)) == "UNSTABLE"
    assert strat.record({}, _result()) == "SAMPLED"
