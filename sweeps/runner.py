"""Trial execution: launch one training run with config overrides, block
until it finishes, and extract the real success + stability metrics.

This is the shared execution machinery every search strategy uses - the
"run one point in the space and score it on the REAL metric" primitive. The
launch/poll/extract logic is ported from scripts/hillclimb_rewards.py (a
proven pattern) and generalized: any task's train_flag, any success/
stability tag, config-override instead of source edits.

Plain-subprocess orchestration: this module does NOT import isaaclab/
isaacsim. It launches isaaclab.sh as a subprocess per trial. Import and use
it from a plain ``python3`` script (scripts/sweep.py), not from under
isaaclab.sh.
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass

from .spaces import TaskSpace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAACLAB_SH = "/home/saps/IsaacLab/isaaclab.sh"

POLL_INTERVAL_S = 15
DEFAULT_TIMEOUT_S = 30 * 60  # generous; a 300-iter diagnostic is ~3-5 min

# Automatic instability reject gate. A genuinely divergent value-function
# loss is orders of magnitude above this repo's known-benign transient
# spikes (Experiment 15's documented benign spike was ~17.66); a real
# divergence runs into the hundreds/thousands or NaN/inf. This default is
# deliberately conservative - it hard-rejects unmistakable divergence
# without discarding benign transients (the exact judgment call the original
# hillclimb spec left to a human). Tunable per-batch via --vf_max_reject.
DEFAULT_VF_MAX_REJECT = 1000.0


@dataclass
class TrialResult:
    success_metric: float | None
    stability_metric: float | None
    run_dir: str | None
    returncode: int | None
    completed: bool
    traceback_found: bool
    unstable: bool

    @property
    def errored(self) -> bool:
        return (
            (not self.completed)
            or self.traceback_found
            or self.success_metric is None
            or (isinstance(self.success_metric, float) and math.isnan(self.success_metric))
        )


class TrialRunner:
    def __init__(
        self,
        num_envs: int = 4096,
        max_iterations: int = 300,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        vf_max_reject: float = DEFAULT_VF_MAX_REJECT,
        work_dir: str = "logs/sweeps",
    ):
        self.num_envs = num_envs
        self.max_iterations = max_iterations
        self.timeout_s = timeout_s
        self.vf_max_reject = vf_max_reject
        self.work_dir = os.path.join(REPO_ROOT, work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

    def run(self, space: TaskSpace, overrides: dict, trial_tag: str) -> TrialResult:
        """Run one trial. ``overrides`` is a flat {override_key: value} dict
        (empty for a baseline trial). ``trial_tag`` names this trial's
        override/log files."""
        start_time = time.time()
        overrides_path = os.path.join(self.work_dir, f"{trial_tag}_overrides.json")
        log_path = os.path.join(self.work_dir, f"{trial_tag}.log")
        with open(overrides_path, "w") as f:
            json.dump(overrides, f, indent=2, sort_keys=True)

        proc, log_f = self._launch(space, overrides_path if overrides else None, log_path)
        run_dir, completed, returncode = self._wait(proc, start_time)
        try:
            log_f.close()
        except Exception:
            pass

        traceback_found = self._check_traceback(log_path)
        success_val, stability_max = (None, None)
        if run_dir is not None:
            success_val, stability_max = self._extract(run_dir, space)

        unstable = self._is_unstable(stability_max)
        return TrialResult(
            success_metric=success_val,
            stability_metric=stability_max,
            run_dir=run_dir,
            returncode=returncode,
            completed=completed,
            traceback_found=traceback_found,
            unstable=unstable,
        )

    # -- internals ---------------------------------------------------------

    def _launch(self, space: TaskSpace, overrides_path: str | None, log_path: str):
        log_f = open(log_path, "w")
        cmd = [ISAACLAB_SH, "-p", "scripts/train.py"]
        if space.train_flag:
            cmd.append(space.train_flag)
        cmd += [
            "--num_envs",
            str(self.num_envs),
            "--max_iterations",
            str(self.max_iterations),
            "--headless",
        ]
        if overrides_path is not None:
            cmd += ["--overrides_file", overrides_path]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=env)
        return proc, log_f

    def _wait(self, proc, start_time: float):
        target = f"model_{self.max_iterations - 1}.pt"
        deadline = start_time + self.timeout_s

        def newest_run_dir():
            dirs = [d for d in glob.glob(os.path.join(REPO_ROOT, "logs/train/*/")) if os.path.getmtime(d) >= start_time]
            return max(dirs, key=os.path.getmtime) if dirs else None

        while True:
            run_dir = newest_run_dir()
            if run_dir is not None and os.path.isfile(os.path.join(run_dir, target)):
                return run_dir, True, proc.poll()
            ret = proc.poll()
            if ret is not None:
                time.sleep(2)
                run_dir = newest_run_dir()
                ok = run_dir is not None and os.path.isfile(os.path.join(run_dir, target))
                return run_dir, ok, ret
            if time.time() >= deadline:
                try:
                    proc.kill()
                except Exception:
                    pass
                return run_dir, False, None
            time.sleep(POLL_INTERVAL_S)

    @staticmethod
    def _check_traceback(log_path: str) -> bool:
        try:
            with open(log_path, errors="replace") as f:
                return "Traceback (most recent call last)" in f.read()
        except OSError:
            return False

    def _extract(self, run_dir: str, space: TaskSpace):
        cmd = [
            ISAACLAB_SH,
            "-p",
            os.path.join(REPO_ROOT, "sweeps", "_extract_scalars.py"),
            run_dir,
            space.success_metric,
            space.stability_metric,
        ]
        try:
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            return None, None
        success_val, stability_max = None, None
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 5 and parts[1] == "LAST" and parts[3] == "MAX":
                tag = parts[0]
                try:
                    last_val = float(parts[2])
                    max_val = float(parts[4])
                except ValueError:
                    continue
                if tag == space.success_metric:
                    success_val = last_val
                elif tag == space.stability_metric:
                    stability_max = max_val
        return success_val, stability_max

    def _is_unstable(self, stability_max: float | None) -> bool:
        if stability_max is None:
            return False  # missing metric is an ERROR path, not an UNSTABLE verdict
        if math.isnan(stability_max) or math.isinf(stability_max):
            return True
        return stability_max > self.vf_max_reject
