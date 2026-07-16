#!/usr/bin/env python3
"""Sync a single Isaac Lab training run directory to Google Cloud Storage.

Plain python3, stdlib only (plus `subprocess` calls out to the `gcloud`
CLI) -- this script must run without Isaac Sim's python environment and
without the `tensorboard` package installed. It does NOT extract scalar
metrics from the TensorBoard event file (that needs `tensorboard`, which
needs Isaac's python env) -- `final_metrics` is always written as `null`
in the generated manifest; a future script can backfill it separately.

Usage:
    python3 scripts/sync_run_to_gcs.py \
        --run-dir logs/train_franka_jointdie/2026-07-12_06-56-02 \
        --experiment joint-space-die-lift \
        [--bucket gs://rl-manipulation-hks-runs] \
        [--checkpoints all|final|none] \
        [--backfill] \
        [--dry-run]

Bucket layout produced:
    gs://<bucket>/<experiment>/<variant>/seed<K>/<timestamp>/...

`variant` is derived from the log-root directory name (the parent of
`--run-dir`'s parent, i.e. the `train_franka*` directory), `seed` is
parsed out of `params/agent.yaml`, and `timestamp` is the run directory's
own basename (e.g. `2026-07-12_06-56-02`).
"""

import argparse
import datetime
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# `gcloud` is not always on PATH in non-interactive shells in this
# environment; fall back to the known SDK install location.
_GCLOUD_FALLBACK = str(Path.home() / "google-cloud-sdk" / "bin" / "gcloud")


def find_gcloud() -> str:
    found = shutil.which("gcloud")
    if found:
        return found
    if os.path.isfile(_GCLOUD_FALLBACK) and os.access(_GCLOUD_FALLBACK, os.X_OK):
        return _GCLOUD_FALLBACK
    print(
        f"ERROR: could not find 'gcloud' on PATH or at {_GCLOUD_FALLBACK}",
        file=sys.stderr,
    )
    sys.exit(1)

# Exact mapping from log-root directory basename -> variant name used in
# the GCS bucket layout.
VARIANT_MAP = {
    "train_franka": "ik-cube",
    "train_franka_jointdie": "joint-die",
    "train_franka_jointcube": "joint-cube",
    "train_franka_jointdieheavy": "joint-die-heavy",
    "train_franka_jointdiebig": "joint-die-big",
    "train_franka_jointcubebaked": "joint-cube-baked",
    "train_franka_jointdiemixed": "joint-die-mixed",
    "train_franka_jointdiemid": "joint-die-mid",
    "train_franka_jointdied8std": "joint-die-d8-std",
    "train_franka_jointdied10std": "joint-die-d10-std",
    "train_franka_jointdied12std": "joint-die-d12-std",
}

DEFAULT_BUCKET = os.environ.get("GCS_BUCKET", "gs://rl-manipulation-hks-runs")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Path to a logs/train_franka*/<timestamp>/ run directory")
    p.add_argument("--experiment", required=True, help="Experiment name, e.g. 'joint-space-die-lift'")
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"Destination bucket (default {DEFAULT_BUCKET})")
    ckpt_group = p.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--checkpoints",
        choices=["all", "final", "none"],
        default="all",
        help="Which model_*.pt checkpoints to upload (default: all)",
    )
    ckpt_group.add_argument(
        "--exclude-checkpoints",
        action="store_true",
        help="Shorthand for --checkpoints none",
    )
    p.add_argument(
        "--backfill",
        action="store_true",
        help="Mark manifest.json's git_sha as backfill_sha_approximate=true "
        "(the exact training-time SHA is unknown for pre-existing runs)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would happen without uploading")
    return p.parse_args()


def derive_variant(run_dir: Path) -> str:
    """variant is keyed off the log-root dir name (run_dir's parent)."""
    log_root_name = run_dir.parent.name
    if log_root_name not in VARIANT_MAP:
        raise ValueError(
            f"Unrecognized log-root directory '{log_root_name}' (parent of run-dir). "
            f"Expected one of: {sorted(VARIANT_MAP)}"
        )
    return VARIANT_MAP[log_root_name]


def parse_seed(agent_yaml_path: Path) -> int:
    text = agent_yaml_path.read_text()
    m = re.search(r"^seed:\s*(-?\d+)\s*$", text, re.MULTILINE)
    if not m:
        raise ValueError(f"Could not find 'seed: <int>' in {agent_yaml_path}")
    return int(m.group(1))


def get_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent.parent
        )
        return out.stdout.strip()
    except Exception as e:
        return f"unknown ({e})"


def checkpoint_files(run_dir: Path):
    return sorted(
        run_dir.glob("model_*.pt"),
        key=lambda p: _checkpoint_sort_key(p),
    )


def _checkpoint_sort_key(p: Path):
    m = re.search(r"model_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


def event_files(run_dir: Path):
    return sorted(run_dir.glob("events.out.tfevents.*"))


def build_manifest(run_dir: Path, experiment: str, variant: str, seed: int, timestamp: str, backfill: bool, checkpoints_kept):
    return {
        "experiment": experiment,
        "variant": variant,
        "seed": seed,
        "timestamp": timestamp,
        "git_sha": get_git_sha(),
        "backfill_sha_approximate": bool(backfill),
        "hostname": socket.gethostname(),
        "upload_time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(run_dir),
        "checkpoint_count": len(checkpoints_kept),
        "event_file_names": [p.name for p in event_files(run_dir)],
        "final_metrics": None,
    }


def prune_checkpoints_for_dry_run_report(run_dir: Path, mode: str):
    """Returns (kept, excluded) checkpoint Path lists per --checkpoints mode.

    Does not delete anything -- upload exclusion is handled by rsync's
    --exclude regex for 'none'/'final', not by mutating the run dir.
    """
    all_ckpts = checkpoint_files(run_dir)
    if mode == "all":
        return all_ckpts, []
    if mode == "none":
        return [], all_ckpts
    if mode == "final":
        if not all_ckpts:
            return [], []
        final = all_ckpts[-1]
        return [final], [c for c in all_ckpts if c != final]
    raise ValueError(mode)


def main():
    args = parse_args()
    checkpoints_mode = "none" if args.exclude_checkpoints else args.checkpoints

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run-dir does not exist or is not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    agent_yaml = run_dir / "params" / "agent.yaml"
    if not agent_yaml.is_file():
        print(f"SKIP: missing {agent_yaml} -- not a valid run dir, skipping {run_dir}")
        sys.exit(0)

    evs = event_files(run_dir)
    if not evs:
        print(f"SKIP: no events.out.tfevents.* file found in {run_dir} (empty/incomplete run), skipping")
        sys.exit(0)

    try:
        variant = derive_variant(run_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        seed = parse_seed(agent_yaml)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    timestamp = run_dir.name
    kept_ckpts, excluded_ckpts = prune_checkpoints_for_dry_run_report(run_dir, checkpoints_mode)

    manifest = build_manifest(run_dir, args.experiment, variant, seed, timestamp, args.backfill, kept_ckpts)
    manifest_path = run_dir / "manifest.json"

    dest = f"{args.bucket.rstrip('/')}/{args.experiment}/{variant}/seed{seed}/{timestamp}/"

    print(f"run_dir      = {run_dir}")
    print(f"experiment   = {args.experiment}")
    print(f"variant      = {variant}")
    print(f"seed         = {seed}")
    print(f"timestamp    = {timestamp}")
    print(f"checkpoints  = {checkpoints_mode} ({len(kept_ckpts)} kept, {len(excluded_ckpts)} excluded)")
    print(f"destination  = {dest}")

    if args.dry_run:
        print(f"[dry-run] would write manifest to {manifest_path}")
        print(f"[dry-run] would run: gcloud storage rsync -r {run_dir} {dest} [checkpoint excludes as needed]")
        return

    # Write manifest.json into the run dir so it gets picked up by rsync.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest: {manifest_path}")

    cmd = [find_gcloud(), "storage", "rsync", "-r", str(run_dir), dest]
    if checkpoints_mode == "none":
        cmd += ["--exclude", r".*model_\d+\.pt$"]
    elif checkpoints_mode == "final" and excluded_ckpts:
        # rsync --exclude regex is matched against the path relative to
        # SOURCE; checkpoints sit directly at the run-dir root, so there is
        # no "/" prefix -- match with an optional leading path segment.
        excluded_names = "|".join(re.escape(p.name) for p in excluded_ckpts)
        cmd += ["--exclude", rf"(^|.*/)({excluded_names})$"]

    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: gcloud storage rsync failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"UPLOADED -> {dest}")


if __name__ == "__main__":
    main()
