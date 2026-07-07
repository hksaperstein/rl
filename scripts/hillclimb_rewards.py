"""Fast, unattended reward hill-climbing loop for the AR4 base-proximity task.

Repeatedly mutates one of 6 reward-term weights/thresholds in
`tasks/ar4/pickplace_baseproximity_env_cfg.py`, launches a short diagnostic
training round, checks whether the real success-termination rate
(`Episode_Termination/cube_reached_goal`) improved, and git-commits the
mutation if so or git-reverts it if not. See
`docs/superpowers/specs/2026-07-07-ar4-hillclimb-loop-design.md` for the
full design this implements.

This is a plain orchestration script - it does NOT import isaaclab/isaacsim
itself, it only launches `isaaclab.sh` as a subprocess for each training
round and each TensorBoard scalar-extraction step. Invoke with plain
`python3`, not `isaaclab.sh`:

.. code-block:: bash

    cd ~/projects/rl
    python3 scripts/hillclimb_rewards.py
    python3 scripts/hillclimb_rewards.py --rounds 15 --num_envs 4096 --max_iterations 300

Round 0 is always a BASELINE-only round: it trains on the current,
unmutated file to establish the starting `cube_reached_goal` value before
any mutation is compared against it. No file is mutated and no commit/
revert happens for round 0.

Safety / scope: only ever mutates the 6 named numeric literals below, in
exactly this one file. Never runs `git push` - that stays a Principal
decision after reviewing a whole batch's results.
"""

import argparse
import glob
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAACLAB_SH = "/home/saps/IsaacLab/isaaclab.sh"
ENV_CFG_PATH = "tasks/ar4/pickplace_baseproximity_env_cfg.py"
RESULTS_PATH = "docs/superpowers/plans/2026-07-07-ar4-hillclimb-results.md"

POLL_INTERVAL_S = 15
TRAINING_TIMEOUT_S = 15 * 60

# Tunable parameter registry (round-robin order matches the design spec's
# table exactly - round i targets PARAMS[i % 6]).
PARAMS = [
    {
        "name": "ground_penalty_weight",
        "block": "ground_penalty = RewTerm(",
        "kind": "weight",
        "step_mode": "mult",
        "step": 1.5,
        "bounds": (0.01, 2.0),
    },
    {
        "name": "ground_height_threshold",
        "block": "ground_penalty = RewTerm(",
        "kind": "dict",
        "key": "ground_height_threshold",
        "step_mode": "add",
        "step": 0.005,
        "bounds": (0.005, 0.05),
    },
    {
        "name": "base_proximity_penalty_weight",
        "block": "base_proximity_penalty = RewTerm(",
        "kind": "weight",
        "step_mode": "mult",
        "step": 1.5,
        "bounds": (0.01, 2.0),
    },
    {
        "name": "base_xy_threshold",
        "block": "base_proximity_penalty = RewTerm(",
        "kind": "dict",
        "key": "base_xy_threshold",
        "step_mode": "add",
        "step": 0.02,
        "bounds": (0.02, 0.15),
    },
    {
        "name": "antipodal_grasp_bonus_weight",
        "block": "antipodal_grasp_bonus = RewTerm(",
        "kind": "weight",
        "step_mode": "add",
        "step": 1.0,
        "bounds": (1.0, 10.0),
    },
    {
        "name": "stillness_penalty_weight",
        "block": "stillness_penalty = RewTerm(",
        "kind": "weight",
        "step_mode": "add",
        "step": 1.0,
        "bounds": (1.0, 12.0),
    },
]


# ---------------------------------------------------------------------------
# Block-scoped file mutation
# ---------------------------------------------------------------------------


def _find_block(lines, block_marker):
    """Locate a RewTerm block by its variable-name line, then find the
    closing ')' at the SAME indentation as the opening line. Never a
    whole-file replace - two blocks in this file share an identical
    `weight=0.1,` line, so all field lookups below must be scoped to the
    line range returned here."""
    marker_stripped = block_marker.strip()
    for idx, line in enumerate(lines):
        if line.strip() == marker_stripped:
            indent = len(line) - len(line.lstrip(" "))
            closing = " " * indent + ")"
            for end_idx in range(idx + 1, len(lines)):
                if lines[end_idx].rstrip("\n") == closing:
                    return idx, end_idx, indent
            raise RuntimeError(f"Could not find closing ')' for block {block_marker!r}")
    raise RuntimeError(f"Could not find block start {block_marker!r} in {ENV_CFG_PATH}")


def _field_pattern(param):
    if param["kind"] == "weight":
        return re.compile(r"^(?P<indent>\s*)weight=(?P<val>-?\d+\.?\d*(?:[eE][-+]?\d+)?),\s*$")
    key = param["key"]
    return re.compile(r'^(?P<indent>\s*)"' + re.escape(key) + r'":\s*(?P<val>-?\d+\.?\d*(?:[eE][-+]?\d+)?),\s*$')


def _find_field(lines, start_idx, end_idx, param):
    pattern = _field_pattern(param)
    for i in range(start_idx, end_idx + 1):
        m = pattern.match(lines[i].rstrip("\n"))
        if m:
            return i, m
    raise RuntimeError(f"Could not find field for param {param['name']!r} within block {param['block']!r}")


def read_param_value(param):
    """Read the parameter's CURRENT value live from the file (never an
    in-memory cache)."""
    with open(ENV_CFG_PATH) as f:
        lines = f.readlines()
    start, end, _ = _find_block(lines, param["block"])
    _, m = _find_field(lines, start, end, param)
    return float(m.group("val"))


def format_literal(x):
    """Format a float back into the file's existing literal style (e.g.
    4.0, 0.1, 0.015) - always at least one explicit decimal digit."""
    x = round(x, 6)
    if x == int(x):
        return f"{x:.1f}"
    s = f"{x:.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def mutate_param_value(param, new_value):
    """Block-scoped in-place edit: only the matched field line within this
    param's own RewTerm block is touched, nothing else in the file."""
    with open(ENV_CFG_PATH) as f:
        lines = f.readlines()
    start, end, _ = _find_block(lines, param["block"])
    i, m = _find_field(lines, start, end, param)
    old_line = lines[i]
    prefix = old_line[: m.start("val")]
    suffix = old_line[m.end("val") :]
    lines[i] = prefix + format_literal(new_value) + suffix
    with open(ENV_CFG_PATH, "w") as f:
        f.writelines(lines)


def propose_new_value(param, current):
    """Round-robin coordinate ascent proposal: random direction, clamp to
    bounds - if already at a bound in the chosen direction, force the
    other direction instead of clamping to a no-op."""
    lo, hi = param["bounds"]
    step_mode = param["step_mode"]
    step = param["step"]

    def step_value(base, direction):
        if step_mode == "mult":
            return base * step if direction > 0 else base / step
        return base + step if direction > 0 else base - step

    direction = random.choice((1, -1))
    candidate = step_value(current, direction)
    if candidate < lo or candidate > hi:
        direction = -direction
        candidate = step_value(current, direction)
    candidate = min(max(candidate, lo), hi)
    return round(candidate, 6)


# ---------------------------------------------------------------------------
# Training round: launch, block until complete, extract scalars
# ---------------------------------------------------------------------------


def launch_training(num_envs, max_iterations, log_path):
    log_f = open(log_path, "w")
    cmd = [
        ISAACLAB_SH,
        "-p",
        "scripts/train.py",
        "--baseproximity",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--headless",
    ]
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    return proc, log_f


def wait_for_completion(proc, round_start_time, max_iterations):
    """Real blocking loop (time.sleep in a loop) inside this script's own
    process - polls every POLL_INTERVAL_S for the new run's final
    checkpoint in the newest logs/train/*/ directory created after
    round_start_time. Timeout: TRAINING_TIMEOUT_S; if exceeded, kill the
    process and report failure (caller marks ERROR + revert) rather than
    hanging indefinitely."""
    target_name = f"model_{max_iterations - 1}.pt"
    deadline = round_start_time + TRAINING_TIMEOUT_S

    def newest_run_dir():
        dirs = [d for d in glob.glob("logs/train/*/") if os.path.getmtime(d) >= round_start_time]
        return max(dirs, key=os.path.getmtime) if dirs else None

    while True:
        run_dir = newest_run_dir()
        if run_dir is not None and os.path.isfile(os.path.join(run_dir, target_name)):
            return run_dir, True, proc.poll()

        ret = proc.poll()
        if ret is not None:
            # Process finished (successfully or not) but the checkpoint
            # wasn't there on the check above - give it one short grace
            # window for the final save to land on disk, then declare it.
            time.sleep(2)
            run_dir = newest_run_dir()
            if run_dir is not None and os.path.isfile(os.path.join(run_dir, target_name)):
                return run_dir, True, ret
            return run_dir, False, ret

        if time.time() >= deadline:
            try:
                proc.kill()
            except Exception:
                pass
            return run_dir, False, None

        time.sleep(POLL_INTERVAL_S)


def check_traceback(log_path):
    try:
        with open(log_path, errors="replace") as f:
            return "Traceback (most recent call last)" in f.read()
    except OSError:
        return False


def extract_scalars(run_dir):
    """Extract Episode_Termination/cube_reached_goal's last value and
    Loss/value_function's max value via isaaclab.sh -p -c, reusing this
    repo's established event_accumulator pattern (see
    docs/superpowers/plans/2026-07-07-ar4-experiment15-reward-shaping-implementation.md's
    "Extract full scalar trajectories" step)."""
    glob_pattern = run_dir + "events.out.tfevents.*"
    code = f"""
from tensorboard.backend.event_processing import event_accumulator
import glob
paths = sorted(glob.glob({glob_pattern!r}))
if not paths:
    print("NO_EVENT_FILE")
else:
    path = paths[-1]
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    tags = ["Episode_Termination/cube_reached_goal", "Loss/value_function"]
    for tag in tags:
        if tag in ea.Tags()["scalars"]:
            vals = ea.Scalars(tag)
            print(tag, "LAST", vals[-1].value, "MAX", max(v.value for v in vals))
        else:
            print(tag, "NOT_FOUND")
"""
    try:
        result = subprocess.run(
            [ISAACLAB_SH, "-p", "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return None, None

    cube_last, vf_max = None, None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1] == "LAST" and parts[3] == "MAX":
            tag = parts[0]
            try:
                last_val = float(parts[2])
                max_val = float(parts[4])
            except ValueError:
                continue
            if tag == "Episode_Termination/cube_reached_goal":
                cube_last = last_val
            elif tag == "Loss/value_function":
                vf_max = max_val
    return cube_last, vf_max


# ---------------------------------------------------------------------------
# Git operations (scoped to exactly one file; never push)
# ---------------------------------------------------------------------------


def git_rev_parse_head():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip()


def git_checkout_file():
    subprocess.run(["git", "checkout", "--", ENV_CFG_PATH], cwd=REPO_ROOT, check=True)


def git_commit_file(message):
    subprocess.run(["git", "add", ENV_CFG_PATH], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT, check=True)


# ---------------------------------------------------------------------------
# Results ledger
# ---------------------------------------------------------------------------


def ensure_results_file():
    if os.path.exists(RESULTS_PATH):
        return
    with open(RESULTS_PATH, "w") as f:
        f.write("# AR4 Reward Hill-Climbing Results\n\n")
        f.write(
            "Results log for `scripts/hillclimb_rewards.py`. See "
            "`docs/superpowers/specs/2026-07-07-ar4-hillclimb-loop-design.md` for the full design.\n\n"
        )
        f.write(
            "| round | parameter | old_value | new_value | cube_reached_goal | value_function_max | outcome | timestamp |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|\n")


def append_result_row(round_num, param_name, old_value, new_value, cube_last, vf_max, outcome):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    old_s = "-" if old_value is None else f"{old_value:.6f}"
    new_s = "-" if new_value is None else f"{new_value:.6f}"
    if cube_last is None or (isinstance(cube_last, float) and math.isnan(cube_last)):
        cube_s = "NaN"
    else:
        cube_s = f"{cube_last:.6f}"
    vf_s = "-" if vf_max is None else f"{vf_max:.6f}"
    with open(RESULTS_PATH, "a") as f:
        f.write(f"| {round_num} | {param_name} | {old_s} | {new_s} | {cube_s} | {vf_s} | {outcome} | {ts} |\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_round(i, num_envs, max_iterations, best_cube_reached_goal):
    round_start_time = time.time()
    head_sha = git_rev_parse_head()
    print(f"\n=== Round {i} (HEAD={head_sha[:10]}) ===")

    is_baseline = i == 0
    param = None
    old_value = None
    new_value = None

    if is_baseline:
        print("Baseline round: no mutation, training on the current file as-is.")
    else:
        param = PARAMS[i % len(PARAMS)]
        old_value = read_param_value(param)
        new_value = propose_new_value(param, old_value)
        mutate_param_value(param, new_value)
        print(f"Mutated {param['name']}: {old_value} -> {new_value}")

    log_path = f"/tmp/hillclimb_round_{i}.log"
    proc, log_f = launch_training(num_envs, max_iterations, log_path)
    print(f"Launched training (PID={proc.pid}), log: {log_path}")

    run_dir, success, returncode = wait_for_completion(proc, round_start_time, max_iterations)
    try:
        log_f.close()
    except Exception:
        pass

    traceback_found = check_traceback(log_path)

    cube_last, vf_max = (None, None)
    if run_dir is not None:
        cube_last, vf_max = extract_scalars(run_dir)

    is_error = (
        (not success)
        or traceback_found
        or cube_last is None
        or (isinstance(cube_last, float) and math.isnan(cube_last))
    )

    print(
        f"run_dir={run_dir} success={success} returncode={returncode} "
        f"traceback={traceback_found} cube_reached_goal={cube_last} value_function_max={vf_max}"
    )

    if is_baseline:
        if is_error:
            append_result_row(i, "(baseline)", None, None, cube_last, vf_max, "ERROR")
            print("FATAL: baseline round errored - cannot establish a comparison metric. Stopping.")
            sys.exit(1)
        append_result_row(i, "(baseline)", None, None, cube_last, vf_max, "BASELINE")
        print(f"Baseline cube_reached_goal = {cube_last:.6f}")
        return cube_last

    if is_error:
        outcome = "ERROR"
        git_checkout_file()
    elif cube_last > best_cube_reached_goal:
        outcome = "KEPT"
        message = (
            f"hillclimb round {i}: {param['name']} {old_value}->{new_value}, "
            f"cube_reached_goal {best_cube_reached_goal:.6f}->{cube_last:.6f}"
        )
        git_commit_file(message)
        best_cube_reached_goal = cube_last
    else:
        outcome = "REVERTED"
        git_checkout_file()

    append_result_row(i, param["name"], old_value, new_value, cube_last, vf_max, outcome)
    print(f"Round {i} outcome: {outcome}")
    return best_cube_reached_goal


def main():
    parser = argparse.ArgumentParser(description="Hill-climb AR4 base-proximity reward weights/thresholds.")
    parser.add_argument("--rounds", type=int, default=15, help="Total number of rounds, including round 0 (baseline).")
    parser.add_argument("--num_envs", type=int, default=4096, help="Parallel environments for each training round.")
    parser.add_argument("--max_iterations", type=int, default=300, help="Diagnostic-scale iterations per round.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    ensure_results_file()

    best_cube_reached_goal = None
    for i in range(args.rounds):
        best_cube_reached_goal = run_round(i, args.num_envs, args.max_iterations, best_cube_reached_goal)

    print("\nHillclimb batch complete.")
    print(f"Final best cube_reached_goal: {best_cube_reached_goal}")
    print(f"Results log: {RESULTS_PATH}")
    print("Note: this script never runs `git push` - review the batch and push separately.")


if __name__ == "__main__":
    main()
