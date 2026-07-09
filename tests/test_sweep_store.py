"""Sim-independent unit tests for sweeps/store.py's TrialStore (insert +
the coverage/report queries scripts/sweep_report.py relies on) - pure
python, no isaaclab import needed, run with plain pytest:
`pytest tests/test_sweep_store.py -v`

Uses a real temp-file SQLite db per test (via pytest's tmp_path fixture),
not an in-memory mock, so this genuinely exercises the persistence path.
"""

import json

from sweeps.store import TrialRecord, TrialStore, config_hash


def _make_record(**kw):
    defaults = dict(
        sweep_id="sweep1",
        task="touchgoal",
        strategy="hillclimb",
        param_values={"weight": 10.0},
        overrides={"env.rewards.foo.weight": 10.0},
        baseline_git_sha="abc123",
        num_envs=256,
        max_iterations=300,
        success_metric_tag="Episode_Termination/goal_reached",
        stability_metric_tag="Loss/value_function",
        success_metric=0.5,
        stability_metric=2.0,
        outcome="BASELINE",
        run_dir="/tmp/run1",
        log_path="/tmp/run1.log",
        notes="",
    )
    defaults.update(kw)
    return TrialRecord(**defaults)


# ---------------------------------------------------------------------------
# insert + round-trip
# ---------------------------------------------------------------------------


def test_insert_and_trials_for_task_round_trip(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    rec = _make_record()
    trial_id = store.insert(rec)
    assert trial_id == 1

    rows = store.trials_for_task("touchgoal")
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == trial_id
    assert row["task"] == "touchgoal"
    assert row["strategy"] == "hillclimb"
    assert row["success_metric"] == 0.5
    assert row["stability_metric"] == 2.0
    assert row["outcome"] == "BASELINE"
    assert json.loads(row["param_values_json"]) == {"weight": 10.0}
    assert json.loads(row["overrides_json"]) == {"env.rewards.foo.weight": 10.0}
    store.close()


def test_insert_multiple_returns_increasing_ids(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    id1 = store.insert(_make_record())
    id2 = store.insert(_make_record())
    assert id2 == id1 + 1
    store.close()


def test_data_persists_across_store_reopen(tmp_path):
    """The db is a real file on disk, not held only in the connection - a
    fresh TrialStore instance over the same path sees prior inserts."""
    db_path = str(tmp_path / "sweeps.db")
    store1 = TrialStore(db_path)
    store1.insert(_make_record())
    store1.close()

    store2 = TrialStore(db_path)
    rows = store2.trials_for_task("touchgoal")
    assert len(rows) == 1
    store2.close()


# ---------------------------------------------------------------------------
# report queries
# ---------------------------------------------------------------------------


def test_tasks_returns_distinct_sorted(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    store.insert(_make_record(task="graspgoal"))
    store.insert(_make_record(task="touchgoal"))
    store.insert(_make_record(task="touchgoal"))
    assert store.tasks() == ["graspgoal", "touchgoal"]
    store.close()


def test_trials_for_task_ordered_by_id(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    store.insert(_make_record(notes="first"))
    store.insert(_make_record(notes="second"))
    rows = store.trials_for_task("touchgoal")
    assert [r["notes"] for r in rows] == ["first", "second"]
    store.close()


def test_param_coverage_join_returns_correct_rows(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    store.insert(_make_record(param_values={"weight": 10.0}, success_metric=0.4, outcome="BASELINE"))
    store.insert(_make_record(param_values={"weight": 15.0}, success_metric=0.7, outcome="KEPT"))
    cov = store.param_coverage("touchgoal", "weight")
    assert len(cov) == 2
    by_value = {c["value"]: c for c in cov}
    assert by_value[10.0]["success_metric"] == 0.4
    assert by_value[15.0]["success_metric"] == 0.7
    store.close()


def test_param_coverage_scoped_to_task_and_param_name(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    store.insert(_make_record(task="touchgoal", param_values={"weight": 10.0}))
    store.insert(_make_record(task="graspgoal", param_values={"weight": 99.0}))
    store.insert(_make_record(task="touchgoal", param_values={"std": 0.05}))
    cov = store.param_coverage("touchgoal", "weight")
    assert len(cov) == 1
    assert cov[0]["value"] == 10.0
    store.close()


def test_param_names_distinct_and_sorted(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    store.insert(_make_record(param_values={"weight": 10.0, "std": 0.05}))
    store.insert(_make_record(param_values={"weight": 12.0}))
    assert store.param_names("touchgoal") == ["std", "weight"]
    store.close()


def test_best_trials_excludes_error_and_unstable_and_orders_desc(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    store.insert(_make_record(success_metric=0.9, outcome="ERROR"))  # excluded despite high score
    store.insert(_make_record(success_metric=0.95, outcome="UNSTABLE"))  # excluded
    store.insert(_make_record(success_metric=0.5, outcome="BASELINE"))
    store.insert(_make_record(success_metric=0.8, outcome="KEPT"))
    rows = store.best_trials("touchgoal", limit=10)
    assert [r["success_metric"] for r in rows] == [0.8, 0.5]
    store.close()


def test_best_trials_respects_limit(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
        store.insert(_make_record(success_metric=s, outcome="KEPT"))
    rows = store.best_trials("touchgoal", limit=2)
    assert [r["success_metric"] for r in rows] == [0.5, 0.4]
    store.close()


def test_best_trials_excludes_null_success_metric(tmp_path):
    store = TrialStore(str(tmp_path / "sweeps.db"))
    store.insert(_make_record(success_metric=None, outcome="ERROR"))
    store.insert(_make_record(success_metric=0.6, outcome="KEPT"))
    rows = store.best_trials("touchgoal", limit=10)
    assert len(rows) == 1
    assert rows[0]["success_metric"] == 0.6
    store.close()


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------


def test_config_hash_deterministic_and_sensitive_to_inputs():
    h1 = config_hash("abc123", {"env.rewards.foo.weight": 10.0})
    h2 = config_hash("abc123", {"env.rewards.foo.weight": 10.0})
    h3 = config_hash("abc123", {"env.rewards.foo.weight": 11.0})
    h4 = config_hash("def456", {"env.rewards.foo.weight": 10.0})
    assert h1 == h2
    assert h1 != h3
    assert h1 != h4


def test_config_hash_insensitive_to_dict_key_order():
    h1 = config_hash("abc123", {"a": 1.0, "b": 2.0})
    h2 = config_hash("abc123", {"b": 2.0, "a": 1.0})
    assert h1 == h2


def test_trial_record_config_hash_property_matches_module_function():
    rec = _make_record()
    assert rec.config_hash == config_hash(rec.baseline_git_sha, rec.overrides)
