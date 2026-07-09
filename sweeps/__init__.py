"""Reusable framework for iterating over RL training + reward-function
parameters in a consistent, repeatable, queryable way across experiments.

This is the generalized Tier-2 tool (see CLAUDE.md's Workflow section): it
tunes weights/thresholds and training hyperparameters *within an already-
validated mechanism*, it does not add new reward terms/action spaces/
mechanisms (that is Tier 1, a full hypothesis-gated spec/plan process).

Design doc: docs/superpowers/specs/2026-07-09-ar4-parameter-sweep-framework-design.md
Concept article: kb/wiki/concepts/hyperparameter-registry.md ("Tier 2" section).

Modules:
- spaces:     declarative per-task parameter-space definitions (TASK_SPACES).
- store:      SQLite-backed structured store of every trial ever run.
- strategies: search strategies (hillclimb / random / grid) over a space.
- runner:     launches a training trial via isaaclab.sh with config overrides,
              waits for it, and extracts the real success + stability metrics.

Orchestrator CLI:  scripts/sweep.py
Coverage report:   scripts/sweep_report.py
"""

from .spaces import ParameterSpec, TaskSpace, TASK_SPACES  # noqa: F401
from .store import TrialRecord, TrialStore  # noqa: F401
