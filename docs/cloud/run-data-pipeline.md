# Training-run data pipeline (GCS)

Backs up Isaac Lab training run directories (TensorBoard event files,
`params/`, checkpoints) to Google Cloud Storage, and generates a small
`manifest.json` per run for discoverability. Built 2026-07-12 on the
`franka-panda-pivot` branch as part of the Franka platform pivot; covers
local-machine backfill only. No Isaac Sim involvement in this pipeline --
plain `python3` + the `gcloud` CLI.

## Bucket layout

```
gs://rl-manipulation-hks-runs/<experiment>/<variant>/seed<K>/<timestamp>/
    events.out.tfevents.*
    manifest.json
    model_*.pt              (checkpoints, if --checkpoints all/final)
    git/IsaacLab.diff
    params/agent.yaml
    params/env.yaml
```

This mirrors the local `logs/train_franka*/<timestamp>/` run directory
exactly, just re-rooted under `<experiment>/<variant>/seed<K>/` instead of
the local `train_franka*` log-root name.

`<experiment>` is a caller-supplied label grouping runs by research
question (e.g. `franka-lift-baseline`, `joint-space-die-lift`,
`asset-bisect`). `<variant>` is derived automatically from the local
log-root directory name via a fixed mapping (see
`scripts/sync_run_to_gcs.py`'s `VARIANT_MAP`):

| log-root directory              | variant            |
|----------------------------------|---------------------|
| `train_franka`                   | `ik-cube`           |
| `train_franka_jointdie`          | `joint-die`         |
| `train_franka_jointcube`         | `joint-cube`        |
| `train_franka_jointdieheavy`     | `joint-die-heavy`   |
| `train_franka_jointdiebig`       | `joint-die-big`     |
| `train_franka_jointcubebaked`    | `joint-cube-baked`  |

`seed<K>` is parsed out of the run's `params/agent.yaml` (`seed: <int>`
line). `<timestamp>` is the run directory's own basename (e.g.
`2026-07-12_06-56-02`), same as Isaac Lab already names it locally.

## manifest.json schema

Written into the run directory locally, then uploaded alongside everything
else:

```json
{
  "experiment": "joint-space-die-lift",
  "variant": "joint-die",
  "seed": 42,
  "timestamp": "2026-07-12_06-56-02",
  "git_sha": "c8d5ba874193a39df313c776e8c5b2057f045617",
  "backfill_sha_approximate": true,
  "hostname": "home",
  "upload_time_utc": "2026-07-12T22:06:21Z",
  "run_dir": "/home/saps/projects/rl/logs/train_franka_jointdie/2026-07-12_06-56-02",
  "checkpoint_count": 31,
  "event_file_names": ["events.out.tfevents.1783853803.home.9153.0"],
  "final_metrics": null
}
```

Notes on specific fields:

- `git_sha` / `backfill_sha_approximate`: the script always records the
  *current* `git rev-parse HEAD` at upload time. For every run backfilled
  from pre-existing local logs (`--backfill` flag), the true
  training-time SHA is not recoverable after the fact, so
  `backfill_sha_approximate` is set `true` as an explicit, honest flag --
  do not treat `git_sha` on a backfilled manifest as the exact commit that
  produced that run. Future runs synced live (not via `--backfill`) will
  have `backfill_sha_approximate: false` and a trustworthy `git_sha`.
- `final_metrics`: always `null` in this version. Extracting scalar
  summaries (e.g. final `Episode_Termination/cube_reached_goal` rate)
  requires the `tensorboard` package to parse the event file, which in
  turn requires Isaac Sim's python environment
  (`/home/saps/IsaacLab/_isaac_sim/python.sh`) -- deliberately out of
  scope here so `sync_run_to_gcs.py` can run under bare `python3` with no
  extra dependencies. A follow-up script could populate this field
  separately using Isaac's python env.

## Scripts

- `scripts/sync_run_to_gcs.py` -- syncs one run directory. Plain
  `python3`, stdlib only, shells out to `gcloud storage rsync` for the
  actual upload (idempotent -- safe to re-run). See its own `--help` /
  docstring for the full flag list (`--checkpoints all|final|none`,
  `--backfill`, `--dry-run`, etc).
- `scripts/sync_all_franka_runs.sh` -- thin wrapper that walks every
  `logs/train_franka*/*/` run directory and calls the above with the
  right `--experiment` per log root, continuing past per-run
  failures/skips and printing a summary table at the end.

Both scripts resolve `gcloud` via `PATH` first, falling back to
`~/google-cloud-sdk/bin/gcloud` if not found on `PATH` (that fallback was
needed in this environment -- `gcloud` is not on `PATH` in non-interactive
shells here).

## Backfill run (2026-07-12)

`scripts/sync_all_franka_runs.sh` was run once against all pre-existing
local Franka runs. Result: 22/22 run directories uploaded, 0 skipped, 0
failed. Total bucket size after backfill: ~591.6 MB (`gcloud storage du -s
gs://rl-manipulation-hks-runs/`), split as `franka-lift-baseline`
~146.9MB, `joint-space-die-lift` ~93.4MB, `asset-bisect` ~351.3MB.
Spot-checked checkpoint counts and manifest contents against the local
run dirs for one run per experiment; all matched exactly (see
`.superpowers/sdd/task-gcs-pipeline-report.md` for the raw listings).

## Viewing runs locally via TensorBoard against `gs://`

**Verified: does not currently work.** Tried against the installed
TensorBoard 2.21.0 under Isaac Sim's python environment
(`/home/saps/IsaacLab/_isaac_sim/python.sh -m tensorboard.main --logdir
gs://rl-manipulation-hks-runs/franka-lift-baseline`), two ways:

1. Default (TensorBoard's newer Rust-based "fast" data-loading path):
   fails at startup -- `NoAuthMethod` panic in the GCS auth manager
   (`gcs/auth.rs`). It cannot find a `gcloud` CLI on `PATH` inside that
   process and also cannot reach a metadata/OAuth endpoint (DNS
   resolution failure for the auth handshake specifically, even though
   plain `gcloud storage` commands work fine from a normal shell). No
   Application Default Credentials are configured either
   (`~/.config/gcloud/application_default_credentials.json` does not
   exist) -- `gcloud auth application-default login` was never run
   against this SDK install, only a regular user login.
2. `--load_fast=false` (older Python-based multiplexer/gfile path):
   fails at run-discovery time with `ImportError: Please install gcsfs to
   access Google Storage` -- the `gcsfs` package is not installed in
   Isaac Sim's bundled python environment, and TensorBoard also reports
   "TensorFlow installation not found - running with reduced feature
   set" (no real `tensorflow`, so no native GCS filesystem plugin either).

Both failure modes were reproduced live against the real uploaded
`franka-lift-baseline` data (`/data/runs` returned `[]` in both cases;
processes were killed immediately after confirming the failure, per this
project's one-Isaac-Sim/one-experiment-at-a-time discipline -- though note
neither of these test runs touched Isaac Sim itself).

**Workaround (documented, not yet needed today's task volume): mirror
down locally, then point local TensorBoard at the mirror.**

```bash
mkdir -p /tmp/tb_mirror/franka-lift-baseline
~/google-cloud-sdk/bin/gcloud storage rsync -r \
    gs://rl-manipulation-hks-runs/franka-lift-baseline \
    /tmp/tb_mirror/franka-lift-baseline

/home/saps/IsaacLab/_isaac_sim/python.sh -m tensorboard.main \
    --logdir /tmp/tb_mirror/franka-lift-baseline --port 6007
```

This is exactly the same `gcloud storage rsync` machinery the upload
scripts use, just run in reverse (bucket -> local). It re-uses local
TensorBoard, which is already known to work against `logs/` directly.
Re-running the same `rsync` command periodically keeps the mirror current
without re-downloading unchanged files.

A real fix (installing `gcsfs` + setting up Application Default
Credentials in Isaac's python env, or running plain-python TensorBoard
outside Isaac's bundled interpreter) was not attempted -- out of scope for
this task, flagged here as a candidate follow-up if native `gs://` reads
become worth the setup cost.

## Intended instance-side usage (future work, not built here)

For cloud-instance training runs (once GCP compute is actually in use --
see `project_gcp-parallel-compute-setup` in memory, currently blocked on
GPU quota), the intended pattern is a lightweight sidecar loop running
alongside the training process on the instance:

```bash
while true; do
    python3 scripts/sync_run_to_gcs.py \
        --run-dir "$RUN_DIR" \
        --experiment "$EXPERIMENT" \
        --checkpoints final   # avoid re-uploading every intermediate ckpt every 60s
    sleep 60
done
```

Live syncs (no `--backfill`) would get an honest, trustworthy `git_sha` in
their manifest (no `backfill_sha_approximate` caveat), since the sidecar
runs on the same commit that launched training. This keeps run data
durable against instance preemption/termination without waiting for
training to finish. Not implemented in this task -- flagged here as the
concrete next step once cloud compute is actually running.
