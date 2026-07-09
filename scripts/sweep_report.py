"""Coverage + outcome report over the parameter-sweep store.

Answers "what have we tried, in what ranges, with what outcomes" without
opening the DB and reading it top to bottom. Plain python3 (stdlib sqlite3
only).

    # overview of all tasks in the store
    python3 scripts/sweep_report.py

    # per-parameter coverage + outcome summary for one task
    python3 scripts/sweep_report.py --task touchgoal

    # detail on one parameter: every value tried, its success, outcome
    python3 scripts/sweep_report.py --task touchgoal --param learning_rate

    # top trials by real success metric
    python3 scripts/sweep_report.py --task touchgoal --best 10

    # ad hoc SQL against the store (tables: trials, trial_params)
    python3 scripts/sweep_report.py --sql "SELECT strategy, COUNT(*) FROM trials GROUP BY strategy"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from sweeps.spaces import TASK_SPACES  # noqa: E402
from sweeps.store import TrialStore  # noqa: E402


def _fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def overview(store: TrialStore) -> None:
    tasks = store.tasks()
    if not tasks:
        print("No trials recorded yet.")
        return
    print(f"{'task':<16}{'trials':>8}{'best_success':>14}   {'outcomes'}")
    for task in tasks:
        rows = store.trials_for_task(task)
        outcomes: dict[str, int] = {}
        best = None
        for r in rows:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
            if r["success_metric"] is not None and r["outcome"] not in ("ERROR", "UNSTABLE"):
                best = r["success_metric"] if best is None else max(best, r["success_metric"])
        oc = "  ".join(f"{k}:{v}" for k, v in sorted(outcomes.items()))
        print(f"{task:<16}{len(rows):>8}{_fmt(best):>14}   {oc}")


def task_report(store: TrialStore, task: str) -> None:
    rows = store.trials_for_task(task)
    if not rows:
        print(f"No trials for task {task!r}.")
        return
    space = TASK_SPACES.get(task)
    print(f"== Task {task}: {len(rows)} trial(s) ==")
    scored = [r for r in rows if r["success_metric"] is not None and r["outcome"] not in ("ERROR", "UNSTABLE")]
    if scored:
        best = max(scored, key=lambda r: r["success_metric"])
        print(f"best success_metric = {_fmt(best['success_metric'])} (trial {best['id']}, {best['strategy']})")
        print(f"  param_values: {best['param_values_json']}")
    print()
    print(f"{'parameter':<28}{'n':>4}{'min':>10}{'max':>10}{'best@success':>16}{'coverage':>10}")
    for pname in store.param_names(task):
        cov = store.param_coverage(task, pname)
        values = [c["value"] for c in cov]
        scored_cov = [c for c in cov if c["success_metric"] is not None and c["outcome"] not in ("ERROR", "UNSTABLE")]
        best_c = max(scored_cov, key=lambda c: c["success_metric"]) if scored_cov else None
        vmin, vmax = min(values), max(values)
        coverage = "-"
        if space is not None and pname in space.params_by_name:
            lo, hi = space.param(pname).bounds
            coverage = f"{(vmax - vmin) / (hi - lo) * 100:.0f}%" if hi > lo else "-"
        best_at = f"{_fmt(best_c['value'])}@{_fmt(best_c['success_metric'])}" if best_c else "-"
        print(f"{pname:<28}{len(values):>4}{_fmt(vmin):>10}{_fmt(vmax):>10}{best_at:>16}{coverage:>10}")


def param_detail(store: TrialStore, task: str, param: str) -> None:
    cov = store.param_coverage(task, param)
    if not cov:
        print(f"No trials varied parameter {param!r} for task {task!r}.")
        return
    print(f"== {task} / {param}: {len(cov)} trial(s) ==")
    print(f"{'value':>12}{'success':>12}{'stability':>12}{'outcome':>12}{'strategy':>12}{'trial':>8}")
    for c in cov:
        print(
            f"{_fmt(c['value']):>12}{_fmt(c['success_metric']):>12}{_fmt(c['stability_metric']):>12}"
            f"{c['outcome']:>12}{c['strategy']:>12}{c['trial_id']:>8}"
        )


def best_report(store: TrialStore, task: str, limit: int) -> None:
    rows = store.best_trials(task, limit)
    if not rows:
        print(f"No scored trials for task {task!r}.")
        return
    print(f"== Top {len(rows)} trials for {task} by success_metric ==")
    for r in rows:
        pv = json.loads(r["param_values_json"])
        print(f"  #{r['id']} success={_fmt(r['success_metric'])} [{r['strategy']}] {pv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report parameter-sweep coverage and outcomes.")
    parser.add_argument("--db", default="logs/sweeps/sweeps.db")
    parser.add_argument("--task", default=None)
    parser.add_argument("--param", default=None, help="Detail one parameter (requires --task).")
    parser.add_argument("--best", type=int, default=None, help="Show top-N trials by success (requires --task).")
    parser.add_argument("--sql", default=None, help="Run an arbitrary read-only SQL query and print rows.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    if not os.path.exists(args.db):
        raise SystemExit(f"No sweep store at {args.db} yet - run scripts/sweep.py first.")
    store = TrialStore(args.db)

    if args.sql:
        for row in store.query(args.sql):
            print("\t".join(_fmt(row[k]) for k in row.keys()))
    elif args.param:
        if not args.task:
            raise SystemExit("--param requires --task")
        param_detail(store, args.task, args.param)
    elif args.best is not None:
        if not args.task:
            raise SystemExit("--best requires --task")
        best_report(store, args.task, args.best)
    elif args.task:
        task_report(store, args.task)
    else:
        overview(store)

    store.close()


if __name__ == "__main__":
    main()
