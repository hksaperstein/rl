"""One-off diagnostic (Task 3.5 re-audit, 2026-07-19): re-derives the
sustained-lift-per-env count for already-completed eval runs using the
CURRENT (post-`612ef85`) fixed settle-detection logic in
`scripts/franka_checkpoint_review.py`, applied to already-downloaded raw
`heights_*.npy` arrays. Pure offline reanalysis - no Isaac Sim, no GPU,
no new rollout.

Why this is possible at all: everything `franka_checkpoint_review.py`
computes downstream of its `height_history` array (resting_z via
MIN-over-`EARLY_SETTLE_START:EARLY_SETTLE_END`, gain, max_z,
sustained-lift run-length) is plain numpy/torch math over that array plus
`episode_length_steps` - both already saved to each run's own
`heights_*.npy` / summary `.json`, and both unaffected by the
settle-detection bug being re-derived here. Confirmed by reading
`franka_checkpoint_review.py`'s current source before writing this script
(2026-07-19 dispatch instruction) rather than assuming it.

This is a line-for-line port of `franka_checkpoint_review.py`'s
post-rollout analysis tail (`main()`, from `analysis_end = ...` through
the per-env `sustained_lift` computation) - kept in sync by inspection,
not by import, since the source script requires `isaaclab`/`torch` to
even parse its top-level imports and this diagnostic is meant to run on
a plain Python + numpy environment (e.g. the Pi, which has no Isaac Lab
install and no `torch`).

Used to re-audit Task 3.5's already-reported d8-big/d10-big/d12-big 3x3
grid (`docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
distillation.md`) against raw `.npy`/`.json` pairs synced to
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/eval-
artifacts/joint-die-{d8,d10,d12}-big/seed{42,123,7}/` - see
`ROADMAP.md`'s "Task 3.5 re-audit" entry (2026-07-19) for the full result
(8/9 cells unchanged with wide margin; d12-big seed123 corrected 4/8 ->
8/8) and `kb/wiki/experiments/unified-multi-die-specialist-
distillation.md`'s matching section.

If `franka_checkpoint_review.py`'s analysis-tail logic changes again in
the future, update `EARLY_SETTLE_START`/`EARLY_SETTLE_END`/
`LIFT_HEIGHT_THRESHOLD_M`/`SUSTAINED_LIFT_STEPS` below to match, or this
script will silently reproduce the *previous* fix's behavior rather than
the current one - there's no automated check keeping the two in sync.

.. code-block:: bash

    # gsutil cp each run's heights_*.npy/.json into <base>/<shape>/seed<N>/
    # first, then:
    python3 scripts/_diag_settle_reaudit_task35.py <base_dir>
"""

import glob
import json
import os
import sys

import numpy as np

LIFT_HEIGHT_THRESHOLD_M = 0.04
SUSTAINED_LIFT_STEPS = 25
EARLY_SETTLE_START = 10
EARLY_SETTLE_END = 45


def _max_consecutive_true(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def reanalyze(npy_path: str, json_path: str) -> dict:
    """Re-derive sustained-lift verdicts for one (shape, seed) run's raw
    height trajectory, using the current fixed settle-detection method.
    `episode_length_steps`/`video_length_steps` are read from the run's own
    existing summary JSON (metadata fields untouched by the settle-detection
    bug) rather than re-derived.
    """
    with open(json_path) as f:
        old = json.load(f)
    episode_length_steps = old["episode_length_steps"]
    video_length = old["video_length_steps"]

    height_history = np.load(npy_path)  # (video_length, num_envs)
    num_envs = height_history.shape[1]

    analysis_end = min(episode_length_steps - 1, video_length)
    analysis_history = height_history[:analysis_end]

    resting_window_end = min(EARLY_SETTLE_END, analysis_end)
    resting_z = analysis_history[EARLY_SETTLE_START:resting_window_end].min(axis=0)

    post_settle_start = min(EARLY_SETTLE_START, analysis_end)
    post_settle_history = analysis_history[post_settle_start:]

    gain = post_settle_history - resting_z[None, :]
    max_gain = gain.max(axis=0)
    max_z = post_settle_history.max(axis=0)
    lifted_mask = gain >= LIFT_HEIGHT_THRESHOLD_M

    per_env = {}
    for env_idx in range(num_envs):
        run_len = _max_consecutive_true(lifted_mask[:, env_idx].tolist())
        sustained = run_len >= SUSTAINED_LIFT_STEPS
        per_env[env_idx] = {
            "resting_z_m": float(resting_z[env_idx]),
            "max_z_m": float(max_z[env_idx]),
            "max_gain_m": float(max_gain[env_idx]),
            "run_len": run_len,
            "sustained": sustained,
        }
    n_sustained = sum(1 for v in per_env.values() if v["sustained"])
    return {
        "n_sustained": n_sustained,
        "num_envs": num_envs,
        "per_env": per_env,
        "old_n_sustained": old["envs_with_sustained_lift"],
        "old_post_settle_start": old["post_settle_start_step"],
        "new_post_settle_start": post_settle_start,
    }


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    results = {}
    for shape in ["d8", "d10", "d12"]:
        for seed in ["42", "123", "7"]:
            d = os.path.join(base, shape, f"seed{seed}")
            npys = glob.glob(os.path.join(d, "*.npy"))
            jsons = glob.glob(os.path.join(d, "*.json"))
            if not npys or not jsons:
                print(f"MISSING data for {shape} seed{seed}: npy={npys} json={jsons}")
                continue
            r = reanalyze(npys[0], jsons[0])
            results[(shape, seed)] = r
            print(
                f"{shape:4s} seed{seed:4s}: OLD={r['old_n_sustained']}/{r['num_envs']} "
                f"NEW={r['n_sustained']}/{r['num_envs']}  "
                f"(old post_settle_start={r['old_post_settle_start']}, "
                f"new={r['new_post_settle_start']})"
            )
            if r["old_n_sustained"] != r["n_sustained"]:
                print(f"    *** CHANGED *** per_env detail: {r['per_env']}")

    print("\n=== Corrected 3x3 grid (sustained lift / 8) ===")
    for shape in ["d8", "d10", "d12"]:
        row = []
        for seed in ["42", "123", "7"]:
            r = results.get((shape, seed))
            row.append(f"{r['n_sustained']}/{r['num_envs']}" if r else "MISSING")
        print(f"{shape}-big: " + " | ".join(row))
