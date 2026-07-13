# Cloud training (GCP)

Status: **pipeline PROVEN 2026-07-13** (shakedown attempt 3). Recipe of
record: `docs/cloud/franka-cloud-shakedown.md`; attempt-by-attempt
history: `.superpowers/sdd/task-cloud-shakedown-report.md` (untracked).

## What exists

- Project `rl-manipulation-hks`, billing upgraded from free tier
  (2026-07-12, manual Console action — no CLI path exists for either
  the billing-tier upgrade or quota-increase filing).
- Quota: `GPUS-ALL-REGIONS-per-project = 1` (granted 2026-07-12 after
  the initial 0→4 request was denied; the minimum re-file at 1 was
  approved within hours). Regional us-central1 L4 = 1 on-demand + 1
  preemptible. Net: **one cloud GPU at a time, spot or on-demand** —
  cloud is a second training lane, not parallel fan-out, until a
  quota raise succeeds (odds improve with billing history).
- Proven stack: SPOT `g2-standard-4` + 1x L4, DLVM
  `common-cu129-ubuntu-2204-nvidia-580` image, Isaac Sim 5.1.0 +
  Isaac Lab v2.3.1 via pip (NOT the NGC container), repo shipped by
  `git archive | ssh tar -x`, headless training, GCS sync via
  `scripts/sync_run_to_gcs.py`.
- Verified capacity headroom: ik-cube at 4096 envs used ~2.8GB of the
  L4's 23GB — `num_envs` is not the constraint on this SKU.

## Operational lessons (2026-07-13 shakedown)

- **SPOT preemption is frequent and correlated across zones**: two
  genuine preemptions inside ~1h; after the second, all 9 surveyed
  zones were simultaneously stocked out for g2+L4 spot. Long runs need
  checkpoint-resume tolerance (rsl_rl checkpoints every 50 iters
  survive fine) or on-demand provisioning.
- **Recovery pattern when spot capacity is gone**: snapshot the boot
  disk, run the GPU-free sync from a cheap `e2-standard-2` created off
  the snapshot, then delete everything. Artifacts survive preemption;
  the instance is disposable.
- Install gaps beyond the official docs (all folded into the recipe):
  Python 3.11 via deadsnakes; `flatdict==4.0.1` needs
  `--no-build-isolation` (its failure SILENTLY skips the base isaaclab
  package — check the install actually landed); DLVM's compute-only
  driver flavor lacks Vulkan ICD/GL libs (`libnvidia-gl-580-server`).
  Scope `isaaclab.sh --install rsl_rl` rather than the "all" default.
- Teardown discipline: verify `instances list`, `disks list`,
  `snapshots list` all empty after every session — a TERMINATED spot
  instance still bills for disk.

Related: [[vision-platform]] (GCS run-sync pattern shared),
`scripts/sync_run_to_gcs.py`, `scripts/sync_all_franka_runs.sh`.
