# Cloud training (GCP)

Status: **pipeline PROVEN 2026-07-13** (shakedown attempt 3), **re-verified
end-to-end with zero preemptions 2026-07-14/15** (attempt 4, an
independent re-run). Recipe of record:
`docs/cloud/franka-cloud-shakedown.md`; attempt-by-attempt history:
`.superpowers/sdd/task-cloud-shakedown-report.md` (untracked).

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

## Attempt 4 re-run (2026-07-14/15): zero preemptions, recipe confirmed repeatable

A second full end-to-end shakedown, independent of attempt 3, confirms the
recipe isn't a one-off. Instance `rl-franka-shakedown` fell back to
`us-west1-a` — `us-central1-a/b/c` **and** `us-east1-b/c/d` were all
`ZONE_RESOURCE_POOL_EXHAUSTED` simultaneously, a worse stockout than
attempt 3 saw. ik-cube (4096 envs) trained ~35min uninterrupted to
1500/1500 iterations with **zero SPOT preemptions** this time (attempt 3
hit two) — the checkpoint-resume path went unexercised. Total instance
lifetime 54m47s (~0.913hr). Completion was verified directly from the
downloaded tfevents file (every scalar tag has exactly 1500 data points,
steps 0-1499), not inferred from the `model_1499.pt` checkpoint filename
alone — the instance's own tee'd stdout log appeared to stop at iteration
1493, which cross-checked as a stdout-buffering artifact at process exit,
not a real early stop. Artifacts:
`gs://rl-manipulation-hks-runs/cloud-shakedown/ik-cube/seed42/2026-07-15_01-52-15/`.

## Cost: real per-SKU GCP pricing (2026-07-14/15)

The recipe's earlier "<$1 total" cost claim (2026-07-13) was a
duration-based estimate against the `g2-standard-4` machine-type price
alone. Pulling the actual SKUs from the Cloud Billing Catalog API
(`cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus`) shows
**the L4 GPU is a fully separate billed SKU, not bundled into the
machine-type price**:

- `Spot Preemptible G2 Instance Core`: $0.01277/vCPU-hr
- `Spot Preemptible G2 Instance Ram`: $0.001496/GiB-hr
- `Nvidia L4 GPU attached to Spot Preemptible VMs`: $0.2862/GPU-hr — the
  dominant cost component
- `Balanced PD Capacity` (boot disk): $0.10/GiB-month

For `g2-standard-4` (4 vCPU, 16GB RAM, 1x L4): CPU+RAM = $0.075/hr, GPU =
$0.2862/hr, combined **$0.361/hr**. Attempt 4's total: ~$0.33 compute +
~$0.02 disk (150GB × 0.913hr) ≈ **~$0.35**.

**Billing-console reporting lag**: mid-run, the GCP Billing console's
"Compute" category showed only $0.019 — matching just the persistent-disk
charge, not the CPU/RAM/GPU instance charges (~$0.33) at all. The GPU SKU
charge (the dominant cost) simply hadn't posted through the billing
pipeline yet, hours into the run. This project has no BigQuery billing
export configured (`bq ls` returns empty), so there is no CLI/API path to
an authoritative billed dollar figure — only duration × published-SKU-rate
estimates are possible outside the Console, and the Console itself lags
real usage by hours.

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
  instance still bills for disk (see the per-SKU pricing breakdown above:
  disk is `Balanced PD Capacity` at $0.10/GiB-month, billed independently
  of whether the GPU/compute instance is running).
- **TensorBoard-over-GCS Cloud Run service**: a small, separate but
  related capability — `docker/tensorboard-gcs/` serves this project's
  entire `gs://rl-manipulation-hks-runs` bucket live via a Cloud Run
  deployment (`python:3.12-slim` + `tensorboard`/`gcsfs`, rescans the
  bucket once a minute). Deployed and live since 2026-07-12, confirmed
  still up 2026-07-14/15, at
  `https://tensorboard-937841495611.us-central1.run.app` —
  `--allow-unauthenticated`, i.e. a public URL with no auth gate. Reads
  the bucket the way `sync_run_to_gcs.py` writes to it, so any run synced
  via that script (local or cloud-trained) is browsable there without
  downloading anything locally.

Related: [[vision-platform]] (GCS run-sync pattern shared),
`scripts/sync_run_to_gcs.py`, `scripts/sync_all_franka_runs.sh`.
