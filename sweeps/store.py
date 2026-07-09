"""SQLite-backed structured store of every parameter trial ever run.

Why SQLite (not the old markdown table, not JSONL/CSV): the user explicitly
wants "methods for collecting data on what's been tried (coverage of the
parameter space) and how it relates to arriving at a working solution" -
i.e. genuinely queryable coverage-vs-outcome analysis, not a file read top
to bottom. SQLite gives that with zero new runtime dependency (stdlib
``sqlite3``), a single portable file, real indexed queries and joins, and
safe concurrent-reader access while a sweep is still writing. A markdown
table can't be queried; a flat JSONL/CSV can, but answering "what values of
parameter X have we tried, across all tasks and sweeps, and what success did
each reach" means a group-by/join that SQL expresses directly and a flat
file forces you to reimplement in Python each time.

Two tables:
- ``trials``       - one row per trial (the full parameter *vector* as JSON,
                     the real success metric, the stability metric, outcome,
                     config-identity hash, baseline git sha, scale, timestamp).
- ``trial_params`` - normalized (trial_id, param_name, value) rows, so
                     per-parameter coverage queries are a trivial join.

The DB lives at logs/sweeps/sweeps.db by default and is committed-agnostic:
it is data, not source (the framework never edits/commits source per trial -
see the design doc's config-override decision).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

DEFAULT_DB_PATH = "logs/sweeps/sweeps.db"


def config_hash(baseline_git_sha: str, overrides: dict) -> str:
    """Stable identity of a trial's config: baseline source state + the exact
    override vector. Two trials with the same hash trained the same thing."""
    payload = baseline_git_sha + "|" + json.dumps(overrides, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class TrialRecord:
    sweep_id: str
    task: str
    strategy: str
    param_values: dict[str, float]
    """The FULL parameter vector (every parameter this trial pinned, whether
    or not it differs from baseline), keyed by logical parameter name."""
    overrides: dict[str, float]
    """The flat override-key -> value dict actually handed to train.py."""
    baseline_git_sha: str
    num_envs: int
    max_iterations: int
    success_metric_tag: str
    stability_metric_tag: str
    success_metric: float | None = None
    stability_metric: float | None = None
    outcome: str = "PENDING"
    run_dir: str | None = None
    log_path: str | None = None
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    @property
    def config_hash(self) -> str:
        return config_hash(self.baseline_git_sha, self.overrides)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id             TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    task                 TEXT NOT NULL,
    strategy             TEXT NOT NULL,
    config_hash          TEXT NOT NULL,
    baseline_git_sha     TEXT NOT NULL,
    param_values_json    TEXT NOT NULL,
    overrides_json       TEXT NOT NULL,
    num_envs             INTEGER NOT NULL,
    max_iterations       INTEGER NOT NULL,
    success_metric_tag   TEXT NOT NULL,
    success_metric       REAL,
    stability_metric_tag TEXT NOT NULL,
    stability_metric     REAL,
    outcome              TEXT NOT NULL,
    run_dir              TEXT,
    log_path             TEXT,
    notes                TEXT
);
CREATE TABLE IF NOT EXISTS trial_params (
    trial_id   INTEGER NOT NULL REFERENCES trials(id),
    param_name TEXT NOT NULL,
    value      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_task ON trials(task);
CREATE INDEX IF NOT EXISTS idx_trials_sweep ON trials(sweep_id);
CREATE INDEX IF NOT EXISTS idx_trial_params_name ON trial_params(param_name);
"""


class TrialStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def insert(self, rec: TrialRecord) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO trials (
                sweep_id, created_at, task, strategy, config_hash, baseline_git_sha,
                param_values_json, overrides_json, num_envs, max_iterations,
                success_metric_tag, success_metric, stability_metric_tag, stability_metric,
                outcome, run_dir, log_path, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.sweep_id,
                rec.created_at,
                rec.task,
                rec.strategy,
                rec.config_hash,
                rec.baseline_git_sha,
                json.dumps(rec.param_values, sort_keys=True),
                json.dumps(rec.overrides, sort_keys=True),
                rec.num_envs,
                rec.max_iterations,
                rec.success_metric_tag,
                rec.success_metric,
                rec.stability_metric_tag,
                rec.stability_metric,
                rec.outcome,
                rec.run_dir,
                rec.log_path,
                rec.notes,
            ),
        )
        trial_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO trial_params (trial_id, param_name, value) VALUES (?,?,?)",
            [(trial_id, name, float(val)) for name, val in rec.param_values.items()],
        )
        self.conn.commit()
        return trial_id

    # ---- read-side helpers used by scripts/sweep_report.py ----------------

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params))

    def tasks(self) -> list[str]:
        return [r["task"] for r in self.query("SELECT DISTINCT task FROM trials ORDER BY task")]

    def trials_for_task(self, task: str) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM trials WHERE task = ? ORDER BY id", (task,))

    def param_coverage(self, task: str, param_name: str) -> list[sqlite3.Row]:
        return self.query(
            """
            SELECT tp.value AS value, t.success_metric AS success_metric,
                   t.stability_metric AS stability_metric, t.outcome AS outcome,
                   t.id AS trial_id, t.strategy AS strategy
            FROM trial_params tp JOIN trials t ON t.id = tp.trial_id
            WHERE t.task = ? AND tp.param_name = ?
            ORDER BY tp.value
            """,
            (task, param_name),
        )

    def param_names(self, task: str) -> list[str]:
        return [
            r["param_name"]
            for r in self.query(
                """
                SELECT DISTINCT tp.param_name AS param_name
                FROM trial_params tp JOIN trials t ON t.id = tp.trial_id
                WHERE t.task = ? ORDER BY tp.param_name
                """,
                (task,),
            )
        ]

    def best_trials(self, task: str, limit: int = 10) -> list[sqlite3.Row]:
        return self.query(
            """
            SELECT * FROM trials
            WHERE task = ? AND success_metric IS NOT NULL
              AND outcome NOT IN ('ERROR', 'UNSTABLE')
            ORDER BY success_metric DESC LIMIT ?
            """,
            (task, limit),
        )

    def close(self) -> None:
        self.conn.close()
