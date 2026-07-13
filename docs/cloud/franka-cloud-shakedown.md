# Franka cloud training shakedown — recipe (PROVEN end-to-end, 2026-07-13)

Status as of 2026-07-13 (attempt 3, quota granted): this recipe has been
**executed end-to-end and proven** on a live GCP L4 SPOT instance —
instance created → Isaac Sim 5.1.0 + Isaac Lab v2.3.1 installed → repo
shipped via `git archive` → training ran to 1200/1500 iterations (80%,
past two SPOT preemptions) → all checkpoints/logs synced to GCS →
instance and snapshot deleted with zero resources left running. Five
concrete corrections to the recipe below were found empirically and are
folded into the relevant sections (Python 3.11 not preinstalled, a
`flatdict` build-isolation failure that silently skips the base
`isaaclab` package, `isaaclab.sh --install`'s default "all" frameworks
pulling an unpinned torch that self-corrects, missing graphics/Vulkan
libraries on the DLVM image, and `git archive`'s repo copy having no
`.git` dir so `sync_run_to_gcs.py`'s `git_sha` field reads "unknown").
See `.superpowers/sdd/task-cloud-shakedown-report.md`'s "Attempt 3"
section for the full narrative, including two SPOT preemptions and the
non-GPU-instance-for-sync recovery technique used when GPU capacity
became unavailable across every surveyed zone simultaneously.

Prior blocked attempts (billing-tier flag, then `GPUS_ALL_REGIONS=0`
quota) are preserved below for history — both are now resolved (billing
upgraded 2026-07-12; quota granted 2026-07-12T23:09Z).

## Blocker (retry, current): global `GPUS_ALL_REGIONS` quota is 0

```
gcloud compute project-info describe --format="json(quotas)" | grep -A2 GPUS_ALL_REGIONS
{'limit': 0.0, 'metric': 'GPUS_ALL_REGIONS', 'usage': 0.0}
```

This is a **project-wide cap independent of the per-region
`NVIDIA_L4_GPUS` quota** (which is already `1.0` in `us-central1` — that
regional quota alone is not sufficient; the global cap is evaluated too
and is stricter at `0`). Confirmed empirically across ~20
`instances create` attempts (SPOT, `g2-standard-4`, 1x `nvidia-l4`) spread
across `us-central1-a/b/c`, `us-east1-b/c/d`, `us-east4-a/c`,
`us-west1-a/b/c`, `us-west4-a/c`: every zone that had actual SPOT capacity
available returned the quota error below; zones with no capacity returned
`ZONE_RESOURCE_POOL_EXHAUSTED` (stockout) instead, before the quota check
was ever reached — that stockout-vs-quota split is *why* the error looked
inconsistent across zones at first, not evidence the quota is
inconsistently enforced.

```
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - Quota 'GPUS_ALL_REGIONS' exceeded.  Limit: 0.0 globally.
	metric name = compute.googleapis.com/gpus_all_regions
	limit name = GPUS-ALL-REGIONS-per-project
	limit = 0.0
	dimensions = global: global
Try your request in another zone, or view documentation on how to increase quotas: https://cloud.google.com/compute/quotas.
```

**Unblock path**: request a `GPUS_ALL_REGIONS` quota increase (1 is
enough) via the Cloud Console —
https://console.cloud.google.com/iam-admin/quotas?project=rl-manipulation-hks
— filter to `GPUS_ALL_REGIONS`, Compute Engine API. No `gcloud` CLI/API
path submits this from a non-interactive session; it's a Console-UI
action, same shape as the billing-tier upgrade below. Once granted,
re-run the instance-create command below verbatim (try
`us-central1-a/b/c` first per the original plan; if all three are
stockout-exhausted, the zones already surveyed above
(`us-east1-b/c`, `us-west1-a/b/c`, `us-west4-c`) are confirmed to have had
capacity as of this retry and are reasonable fallbacks).

## Blocker (first attempt, resolved): billing account free-tier flag

```
gcloud compute instances create ... \
  --accelerator=type=nvidia-l4,count=1 ...

ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - Your billing account is currently in the free tier where non-TPU accelerators are not available. Please upgrade to a paid billing account as described here: https://cloud.google.com/free/docs/gcp-free-tier#how-to-upgrade
```

Billing account `014C1A-0F2833-2BC2C4` ("My Billing Account") has
`billingEnabled: true` and is linked to project `rl-manipulation-hks`, but
GCP separately flags it as "free tier" for accelerator provisioning — a
different, human-verification-gated flag than "has a payment method."
No `gcloud`/API path resolves this (checked `gcloud billing accounts
--help` GA/alpha/beta — no upgrade subcommand exists). **User must
manually complete the upgrade flow in the Cloud Console** at the URL
above before any GPU instance (spot or on-demand, L4 or otherwise) can be
created on this account. This blocker fires before GPU quota
(`GPUS_ALL_REGIONS`, `NVIDIA_L4_GPUS`, etc.) is even evaluated, so the
spot-vs-`GPUS_ALL_REGIONS` question this task was meant to answer
empirically is still open.

## Instance creation (proven — verbatim command used successfully 2026-07-13)

```bash
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
gcloud compute instances create rl-franka-shakedown \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True \
  --scopes=storage-rw \
  --maintenance-policy=TERMINATE
```

Notes on choices, re-derived live (the task's original ground facts named
an image family, `common-cu12x-debian-11`, that no longer exists):
- Available Deep Learning VM image families today (`gcloud compute images
  list --project deeplearning-platform-release`, filtered to CUDA/pytorch
  families): `common-cu129-ubuntu-2204-nvidia-580`,
  `common-cu129-ubuntu-2404-nvidia-580`,
  `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`,
  `pytorch-2-9-cu129-ubuntu-2404-nvidia-580`. Chose the Ubuntu 22.04
  `common` (non-pytorch-specific) family since Isaac Sim installs its own
  pinned torch (see below) and Ubuntu 22.04 is NVIDIA's explicitly
  documented supported OS for Isaac Sim 5.1 GCP deployments (see below).
  These images already bundle NVIDIA driver 580 (the `-nvidia-580`
  suffix), so `install-nvidia-driver=True` is likely a no-op safety net
  here, not empirically confirmed.
- `us-central1-a/b/c` all list `nvidia-l4` as an available accelerator
  type (`gcloud compute accelerator-types list`); try `-a` first, fall
  back to `-b`/`-c` on a spot-capacity `ZONE_RESOURCE_POOL_EXHAUSTED`
  error if it occurs.
- Regional quota confirmed present before the billing block was hit:
  `NVIDIA_L4_GPUS: 1.0`, `PREEMPTIBLE_NVIDIA_L4_GPUS: 1.0`,
  `COMMITTED_NVIDIA_L4_GPUS: 1.0` (`gcloud compute regions describe
  us-central1`).
- `default-allow-ssh` firewall rule already exists (0.0.0.0/0, tcp:22) —
  no firewall setup needed before `gcloud compute ssh`.
- Fall back to `g2-standard-8` only if Isaac OOMs on the 16GB host RAM of
  `g2-standard-4` (not observed — never reached boot).

NVIDIA's own GCP deployment doc
(https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_advanced_cloud_setup_gcp.html)
explicitly lists `nvidia-l4` + `g2-standard-4 or better` + `Ubuntu 22.04
LTS` as a supported/blessed configuration — this combination was not
picked blind.

## Isaac Sim 5.1 + Isaac Lab install — pip path (PROVEN 2026-07-13, corrections below)

Sourced from NVIDIA's official docs (2026-07-12; re-verified live against
the `v2.3.1` **tag's own** docs on 2026-07-13, not `main` — no drift
found, the command shape below matches the tag verbatim):
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
- `docs/source/setup/installation/pip_installation.rst` in the `v2.3.1` tag itself

```bash
# System graphics/Vulkan libraries — NOT preinstalled on the
# common-cu129-ubuntu-2204 DLVM image. Its NVIDIA driver install is the
# compute-only flavor (libnvidia-compute-580-server); Isaac Sim's RTX
# renderer needs libnvidia-gl (OpenGL/Vulkan ICD) even fully --headless,
# or you get `vkCreateInstance failed: ERROR_INCOMPATIBLE_DRIVER` and the
# process appears to hang indefinitely during scene construction. Match
# the -580-server suffix to whatever driver version nvidia-smi reports.
sudo apt-get update -y
sudo apt-get install -y libgl1 libglx-mesa0 libegl1 libnvidia-gl-580-server \
    vulkan-tools libglu1-mesa libxt6 tmux cmake build-essential

# Isaac Sim 5.1.0 requires Python 3.11 (not 3.10 — that was 4.5.x).
# NOT preinstalled on this image (ships 3.10.12) — deadsnakes PPA needed.
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv ~/isaac-venv
source ~/isaac-venv/bin/activate

# Accept the EULA non-interactively (required on a headless/non-interactive
# shell — otherwise the first `import isaacsim` blocks waiting on stdin).
export OMNI_KIT_ACCEPT_EULA=YES

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.1 IsaacLab
cd IsaacLab

# Workaround for a build-isolation bug in one transitive dependency
# (flatdict==4.0.1, pulled in by the base isaaclab package) BEFORE
# running --install: its isolated build env resolves a setuptools that
# doesn't expose pkg_resources, failing with
# `ModuleNotFoundError: No module named 'pkg_resources'`. --install's own
# find-exec loop over source/* extensions does NOT abort on this failure
# and the script still exits 0, so the base `isaaclab` package (isaaclab.envs,
# isaaclab.controllers, etc.) silently never installs unless this is
# pre-empted:
pip install --no-build-isolation flatdict==4.0.1

# Scope to rsl_rl only (the only framework this repo uses) instead of the
# default "all" -- "all" also installs rl_games/skrl/robomimic, whose
# combined dependency resolution pulls a newer, unpinned torch/torchvision
# (a full separate CUDA-13 wheel stack) that silently uninstalls the
# torch==2.7.0+cu128 pinned above. isaaclab.sh DOES self-correct this at
# the very end of an "all" install (its own final step re-pins torch/
# torchvision from the pip cache) -- so "all" is not actually broken, just
# slower and briefly alarming to watch live. Scoping to rsl_rl avoids the
# multi-GB churn entirely:
./isaaclab.sh --install rsl_rl

# Verify the base isaaclab package actually installed (its own failure,
# per the flatdict issue above, does not surface as a nonzero exit code):
python -c "import isaaclab" 2>&1 | grep -q "No module named 'isaaclab'" && \
    echo "FAILED: base isaaclab package missing, see flatdict fix above" || \
    echo "OK (a 'No module named omni' error here is normal/expected --
          isaaclab.envs etc. only import inside an AppLauncher-bootstrapped
          process, not at bare python -c time)"
```

This deliberately avoids the NGC container path
(`nvcr.io/nvidia/isaac-sim:5.1.0`, used by this repo's own
`docker/Dockerfile.base` + `docker/.env.base`) — the pip path needs no NGC
API key. Branch pinned to `v2.3.1` to match the local Isaac Lab install
(`/home/saps/IsaacLab/VERSION`); the doc URL above is IsaacLab's `main`
branch tree (no version-pinned doc tree was found), so diff the pip
command shape against `v2.3.1`'s own docs/install script once cloned, in
case it changed.

Requirements-page notes relevant to the chosen VM: driver **580.65.06**
minimum (the image's driver 580 build meets this — exact patch version
not yet confirmed against the live instance), GLIBC 2.35+ (Ubuntu 22.04
ships 2.35), 16GB VRAM minimum (L4 has 24GB). The requirements table's
"GPUs without RT Cores (A100, H100) are not supported" caveat does not
exclude the L4 — L4 is Ada Lovelace (AD104) with 3rd-gen RT cores, same
generation as the RTX 40-series named in the table (an initial web-search
synthesis incorrectly claimed L4 lacks RT cores; that claim is wrong and
contradicted by NVIDIA's own GCP deployment doc recommending the L4).

RL dependency pin (from `/home/saps/IsaacLab/source/isaaclab_rl/setup.py`,
installed automatically by `./isaaclab.sh --install`'s `rsl_rl` extra):
`rsl-rl-lib==3.0.1`, `onnxscript>=0.5`.

## Training command (PROVEN 2026-07-13 — unchanged from local, add `--headless`)

```bash
cd ~/rl   # repo checked out via `git archive HEAD | ssh ... tar -x`, not a git clone (private repo, no deploy key)
tmux new-session -d -s train
tmux send-keys -t train '
source ~/isaac-venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/train_franka.py --variant ik-cube --num_envs 4096 \
    --max_iterations 1500 --headless 2>&1 | tee ~/train_franka_cloud.log
' Enter
```
Always launch inside `tmux` (or `nohup`) — an SSH disconnect must not
kill the run. `--num_envs 4096` did **not** OOM the L4's 24GB VRAM (GPU
memory usage stayed ~2.8GB/23GB throughout a real run) — no need to
halve to 2048.

**SPOT preemption is real and should be planned for, not just
theoretically possible**: attempt 3 hit two preemptions in about an hour
of GPU uptime. `save_interval=50` in `tasks/franka/agents/rsl_rl_ppo_cfg.py`
means a checkpoint always exists within 50 iterations of any interruption
— on restart, resume with `--checkpoint <path/to/model_N.pt>
--max_iterations 1500` (absolute target, not "N more"; see
`train_franka.py`'s own docstring). If the zone (or, as happened in
attempt 3, *every* surveyed zone simultaneously) is SPOT-stocked-out and
restart/recreate isn't landing within a few minutes, don't keep polling
indefinitely — either accept the run at its last checkpoint (see the
non-GPU-sync technique below) or fall back to a shorter run per the
mission's own "300 iterations proves the pipeline" bar.

## Sync + cleanup (PROVEN 2026-07-13, see docs/cloud/run-data-pipeline.md)

```bash
python3 scripts/sync_run_to_gcs.py \
    --run-dir logs/train_franka/<timestamp> \
    --experiment cloud-shakedown

# then, from the local machine, regardless of outcome:
gcloud compute instances delete rl-franka-shakedown --zone=us-central1-a --quiet
```

Note: `sync_run_to_gcs.py`'s `git_sha` manifest field will read
`"unknown (...)"` for any run whose repo was shipped via `git archive`
(no `.git` dir present to `git rev-parse HEAD` against) — this is
graceful, not a bug, just not populated. If you need it recorded, note
the local `git rev-parse HEAD` output separately at ship time.

**If the GPU instance is unreachable when it's time to sync** (SPOT
preemption + capacity crunch, as in attempt 3): `sync_run_to_gcs.py`
needs **no GPU and no Isaac Sim** — it's plain `python3` + the `gcloud`
CLI (see its own docstring). Snapshot the (stopped/terminated) instance's
boot disk, then create a small non-GPU instance from that snapshot
(`--machine-type=e2-standard-2`, no `--accelerator`) — this sidesteps GPU
capacity contention entirely since it's a completely different, plentiful
SKU:
```bash
gcloud compute disks snapshot rl-franka-shakedown --zone=<zone> \
    --snapshot-names=rl-franka-shakedown-snap1
gcloud compute instances create rl-franka-sync --zone=<any-zone> \
    --machine-type=e2-standard-2 \
    --create-disk=boot=yes,auto-delete=yes,size=150GB,type=pd-balanced,\
source-snapshot=projects/<project>/global/snapshots/rl-franka-shakedown-snap1 \
    --scopes=storage-rw
# ssh in, run sync_run_to_gcs.py exactly as above against the restored
# ~/rl/logs/train_franka/<timestamp> dir, then delete both the sync
# instance AND the original (terminated) GPU instance AND the snapshot.
```

## Quota / billing status: both prior blockers resolved

Billing-tier free-tier flag: resolved 2026-07-12 (user completed the
Console upgrade flow). `GPUS_ALL_REGIONS` project quota: resolved
2026-07-12T23:09Z (`GPUS-ALL-REGIONS-per-project` `grantedValue=1
APPROVED`). Regional `NVIDIA-L4-GPUS` quota of `1.0` is granted in
`us-central1`, `us-east1`, `us-west1`, and `us-west4` (confirmed
2026-07-13) — quota is not expected to block a future run in any of
these regions; **zone-level SPOT capacity (stockout) is the only
remaining friction**, and it can affect multiple zones/regions
simultaneously (observed 2026-07-13: 8/8 surveyed zones stocked out at
once for `g2-standard-4`+`nvidia-l4`).

## Quota state (2026-07-12 night, controller update)

Empirical (retry attempt, ~20 creates across 11 zones): **SPOT does NOT
bypass `GPUS_ALL_REGIONS`** — every zone with real capacity rejected on
the global quota; stockout zones failed earlier on
ZONE_RESOURCE_POOL_EXHAUSTED, which made errors look inconsistent but
is just check ordering.

The original 0→4 requests (all three, filed via Cloud Quotas API
~21:33Z) were **denied** — the classic zero-history/new-paid-account
pattern. Re-filed `GPUS-ALL-REGIONS-per-project` at the minimum
(preferred=1, matching the already-granted regional L4 quota of 1) at
~04:1xZ; reconciling. If denied again: Console quota page + a support
case are the escalation path, and a few days of billing history
(non-GPU usage counts) materially improves approval odds.
