# Franka cloud training shakedown — recipe (BLOCKED, not yet run)

Status as of 2026-07-12: this recipe has **not been executed end-to-end**.
The first live attempt (`franka-panda-pivot` branch, cloud-shakedown task)
was blocked at instance creation by a billing-account-level restriction,
before any GPU quota, image-boot, or Isaac-install step could be tested.
See `.superpowers/sdd/task-cloud-shakedown-report.md` for the full report.
This doc records the intended recipe (verified against live `gcloud`
output + NVIDIA's official docs, where noted) so the retry doesn't have to
re-derive it.

## Blocker: billing account free-tier flag

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

## Instance creation (verified command shape; blocked before execution completed)

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

## Isaac Sim 5.1 + Isaac Lab install — pip path (researched, NOT yet run live)

Sourced from NVIDIA's official docs (2026-07-12; verify against the live
page again at execution time in case it moves):
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

```bash
# Isaac Sim 5.1.0 requires Python 3.11 (not 3.10 — that was 4.5.x).
python3.11 -m venv ~/isaac-venv
source ~/isaac-venv/bin/activate

# Accept the EULA non-interactively (required on a headless/non-interactive
# shell — otherwise the first `import isaacsim` blocks waiting on stdin).
export OMNI_KIT_ACCEPT_EULA=YES

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.1 IsaacLab
cd IsaacLab
./isaaclab.sh --install
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

## Training command (unchanged from local, add `--headless`)

```bash
cd ~/rl   # repo checked out via `git archive HEAD | ssh ... tar -x`, not a git clone (private repo, no deploy key)
python scripts/train_franka.py --variant ik-cube --num_envs 4096 \
    --max_iterations 1500 --headless 2>&1 | tee train_franka_cloud.log
```

Halve `--num_envs` to 2048 and document if 4096 OOMs the L4's 24GB VRAM
(not observed — never reached this step).

## Sync + cleanup (unchanged, see docs/cloud/run-data-pipeline.md)

```bash
python3 scripts/sync_run_to_gcs.py \
    --run-dir logs/train_franka/<timestamp> \
    --experiment cloud-shakedown

# then, from the local machine, regardless of outcome:
gcloud compute instances delete rl-franka-shakedown --zone=us-central1-a --quiet
```

## Next step

Once the user completes the Cloud Console billing upgrade
(https://cloud.google.com/free/docs/gcp-free-tier#how-to-upgrade), re-run
the instance-creation command above verbatim as the first retry step —
that single command will also finally answer the spot-vs-`GPUS_ALL_
REGIONS` empirical question this task set out to test.
