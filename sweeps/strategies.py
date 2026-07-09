"""Search strategies over a TaskSpace: hillclimb, random, grid.

All three share one driver interface so scripts/sweep.py runs them the same
way and records them into the same store:

    strat = SomeStrategy(space, ...)
    while (pv := strat.next_trial()) is not None:
        result = runner.run(space, resolve_overrides(space, pv), tag)
        store.insert(...)
        strat.record(pv, result)

- ``next_trial()`` returns the next {param_name: value} vector to run, or
  ``None`` when the strategy is exhausted. An empty dict ``{}`` means a
  baseline trial (no overrides).
- ``record(param_values, result)`` feeds the trial's outcome back. Only
  hillclimb uses it (greedy adoption); random/grid ignore it.

Adoption/greedy logic judges ONLY on the real success metric and treats an
errored or unstable trial as non-adoptable, never as an improvement -
matching this project's hard rule that a shaped scalar improving is not
evidence, and a value-function divergence is an automatic reject.
"""

from __future__ import annotations

import itertools
import math
import random

from .spaces import ParameterSpec, TaskSpace


def _random_value(spec: ParameterSpec, rng: random.Random) -> float:
    lo, hi = spec.bounds
    if spec.scale == "log":
        value = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    else:
        value = rng.uniform(lo, hi)
    return spec.quantize(value)


def _grid_values(spec: ParameterSpec, n_points: int) -> list[float]:
    lo, hi = spec.bounds
    if n_points < 2:
        return [spec.quantize(lo)]
    if spec.scale == "log":
        lg_lo, lg_hi = math.log(lo), math.log(hi)
        raw = [math.exp(lg_lo + (lg_hi - lg_lo) * i / (n_points - 1)) for i in range(n_points)]
    else:
        raw = [lo + (hi - lo) * i / (n_points - 1) for i in range(n_points)]
    # De-dup after quantization (integer params can collapse points).
    seen: list[float] = []
    for v in raw:
        q = spec.quantize(v)
        if q not in seen:
            seen.append(q)
    return seen


class HillclimbStrategy:
    """Single-parameter greedy coordinate ascent (the original
    scripts/hillclimb_rewards.py behavior, generalized to any task/space).
    Round 0 is a baseline; each later round steps ONE parameter (round-robin)
    from the current best-known vector and adopts it only if the real success
    metric strictly improves and the trial is stable."""

    name = "hillclimb"

    def __init__(self, space: TaskSpace, rounds: int, param_names: list[str] | None = None, seed: int = 0):
        self.space = space
        self.rounds = rounds
        self.rng = random.Random(seed)
        self.param_names = param_names or [p.name for p in space.parameters]
        self.best_values: dict[str, float] = {}  # param -> adopted value (baseline if absent)
        self.best_success: float | None = None
        self._round = 0
        self._last_proposed: tuple[str, float] | None = None

    def _current(self, name: str) -> float:
        return self.best_values.get(name, self.space.param(name).baseline)

    def _propose(self, spec: ParameterSpec, current: float) -> float:
        lo, hi = spec.bounds

        def step(base: float, direction: int) -> float:
            if spec.step_mode == "mult":
                return base * spec.step if direction > 0 else base / spec.step
            return base + spec.step if direction > 0 else base - spec.step

        direction = self.rng.choice((1, -1))
        candidate = step(current, direction)
        if candidate < lo or candidate > hi:
            candidate = step(current, -direction)
        return spec.quantize(candidate)

    def next_trial(self) -> dict | None:
        if self._round >= self.rounds:
            return None
        if self._round == 0:
            self._round += 1
            self._last_proposed = None
            return {}  # baseline
        name = self.param_names[(self._round - 1) % len(self.param_names)]
        spec = self.space.param(name)
        new_value = self._propose(spec, self._current(name))
        self._last_proposed = (name, new_value)
        # Full vector = current best for all non-baseline params + this step.
        pv = dict(self.best_values)
        pv[name] = new_value
        self._round += 1
        return pv

    def record(self, param_values: dict, result) -> str:
        """Returns the adoption outcome label for this trial."""
        if result.errored:
            return "ERROR"
        if result.unstable:
            return "UNSTABLE"
        success = result.success_metric
        if self._last_proposed is None:
            # Baseline round.
            self.best_success = success
            return "BASELINE"
        if self.best_success is None or success > self.best_success:
            name, value = self._last_proposed
            self.best_values[name] = value
            self.best_success = success
            return "KEPT"
        return "REVERTED"


class RandomStrategy:
    """Random search: each trial samples every varied parameter uniformly
    (linear or log per its scale) within bounds. No adoption - pure coverage
    of the multi-dimensional space."""

    name = "random"

    def __init__(self, space: TaskSpace, n_trials: int, param_names: list[str] | None = None, seed: int = 0):
        self.space = space
        self.n_trials = n_trials
        self.rng = random.Random(seed)
        self.param_names = param_names or [p.name for p in space.parameters]
        self._done = 0

    def next_trial(self) -> dict | None:
        if self._done >= self.n_trials:
            return None
        self._done += 1
        return {name: _random_value(self.space.param(name), self.rng) for name in self.param_names}

    def record(self, param_values: dict, result) -> str:
        if result.errored:
            return "ERROR"
        if result.unstable:
            return "UNSTABLE"
        return "SAMPLED"


class GridStrategy:
    """Grid search over a small set of parameters. ``points_per_param`` sets
    the resolution per axis; the cartesian product is capped at ``max_trials``
    (a full grid over many axes explodes combinatorially - keep the axis set
    small)."""

    name = "grid"

    def __init__(
        self,
        space: TaskSpace,
        param_names: list[str],
        points_per_param: int = 3,
        max_trials: int = 32,
    ):
        self.space = space
        self.param_names = param_names
        axes = [_grid_values(space.param(n), points_per_param) for n in param_names]
        combos = list(itertools.product(*axes))
        self.combos = combos[:max_trials]
        self._i = 0

    def total(self) -> int:
        return len(self.combos)

    def next_trial(self) -> dict | None:
        if self._i >= len(self.combos):
            return None
        combo = self.combos[self._i]
        self._i += 1
        return dict(zip(self.param_names, combo))

    def record(self, param_values: dict, result) -> str:
        if result.errored:
            return "ERROR"
        if result.unstable:
            return "UNSTABLE"
        return "SAMPLED"
