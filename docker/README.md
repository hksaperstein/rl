# Cloud/headless training container

Packages this repo's code + whatever pre-built USD assets the current
task needs into a Docker image, built on Isaac Lab's own official base
image, so `scripts/train.py` runs unmodified on any NVIDIA GPU cloud
instance. Generic to this repo as a whole (a reusable Isaac Lab
manipulation research platform, per CLAUDE.md's North Star), not
hardcoded to any one robot/task — today that's AR4 pick-and-place, but
nothing here assumes that stays true.

## Prerequisites (this workstation, before building)

1. `assets/` must already contain whatever the current task needs (run
   that task's own asset-build script first, e.g. `scripts/build_asset.py`
   for the AR4 work) — the Dockerfile fails the build early with a clear
   error if `assets/` is empty. Already satisfied as of this writing.
2. Docker is installed locally (confirmed: 27.3.1). `nvidia-container-toolkit`
   is **not** currently installed on this workstation, which means the
   image can be *built* here but not *run with GPU access* here — that's
   fine for building and pushing to a registry, but local test-running
   would need `sudo apt install nvidia-container-toolkit` first if ever
   wanted (a real system change, not done as part of this work).

## Build (two steps, from this repo's root)

```bash
# 1. Build Isaac Lab's own base image (one-time, or when its version
#    changes) via Isaac Lab's own official wrapper - not reimplemented
#    here, just reused. (Raw `docker compose --profile base build` does
#    NOT work directly: env_file: only sets the *container's* runtime
#    environment, not build-arg interpolation, so ISAACSIM_VERSION etc.
#    come through blank and the build fails with an invalid FROM. The
#    wrapper handles this correctly. It also prompts for X11 forwarding
#    interactively - answer N non-interactively as below if scripting
#    this.)
cd /home/saps/IsaacLab/docker
echo "N" | python3 container.py start base

# This also tries to START a container with --gpus all, which will fail
# on a workstation without nvidia-container-toolkit installed (confirmed
# on this machine: "could not select device driver nvidia with
# capabilities: [[gpu]]") - that's expected and fine, the image itself
# (isaac-lab-base:latest) still gets built successfully before that
# failure. Confirm with: docker images | grep isaac-lab-base

# 2. Build this repo's image on top of it (verified working - a real
#    24GB image built cleanly from this Dockerfile)
cd /home/saps/projects/rl
docker build -f docker/Dockerfile -t isaaclab-manipulation-rl:latest .
```

## Push to a registry

Pick one per the provider you're deploying to:

```bash
# Docker Hub (works with any provider that can pull public/private images)
docker tag isaaclab-manipulation-rl:latest <dockerhub-username>/isaaclab-manipulation-rl:latest
docker push <dockerhub-username>/isaaclab-manipulation-rl:latest

# AWS ECR
aws ecr create-repository --repository-name isaaclab-manipulation-rl
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag isaaclab-manipulation-rl:latest <account-id>.dkr.ecr.<region>.amazonaws.com/isaaclab-manipulation-rl:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/isaaclab-manipulation-rl:latest

# GCP Artifact Registry
gcloud artifacts repositories create isaaclab-manipulation-rl --repository-format=docker --location=<region>
docker tag isaaclab-manipulation-rl:latest <region>-docker.pkg.dev/<project-id>/isaaclab-manipulation-rl/isaaclab-manipulation-rl:latest
docker push <region>-docker.pkg.dev/<project-id>/isaaclab-manipulation-rl/isaaclab-manipulation-rl:latest
```

RunPod and Lambda Labs both accept a plain Docker Hub image reference
directly when launching a pod/instance — no separate registry push step
beyond Docker Hub needed for those two.

## Run on a cloud GPU instance

All four providers researched (AWS, GCP, RunPod, Lambda) ship NVIDIA GPU
instance images with the container runtime's GPU passthrough already
configured — `--gpus all` (or the provider's equivalent) just works,
unlike this workstation which would need `nvidia-container-toolkit`
installed first.

```bash
docker run --rm -it --gpus all --network host \
    <image-ref> \
    ./isaaclab.sh -p scripts/train.py --pregrasp --num_envs 4096 --headless
```

Getting checkpoints/logs back off the instance: `logs/` is not baked
into the image (excluded via `.dockerignore`, and gitignored in this
repo too) — mount a volume or `docker cp` the container's
`/workspace/rl/logs/` directory out before tearing the instance down,
or `rsync`/`scp` it directly from the instance's host filesystem if not
using `--network host` isolation.

## Per-provider account setup (manual — needs your own payment method/identity)

- **AWS**: console.aws.amazon.com → create account → IAM user with EC2/ECR
  permissions → request GPU instance quota for the target region/instance
  family (G5/G6e quotas often start at 0 and need a support-ticket bump).
- **GCP**: console.cloud.google.com → create project → enable Compute
  Engine API → request GPU quota (also often starts at 0).
- **Lambda Labs**: cloud.lambdalabs.com → sign up → add payment method →
  on-demand instances available immediately, no quota request typically
  needed.
- **RunPod**: runpod.io → sign up → add payment method → Community Cloud
  pods available immediately.

Once you have credentials/API keys for whichever provider(s) you want,
hand them to me (or set them as environment variables / CLI-configured
credentials in this environment) and I can drive the actual instance
launch, image push, and job dispatch from here.

## Publishing to Docker Hub (GitHub Action)

`.github/workflows/docker-publish.yml` builds and pushes
`isaaclab-manipulation-rl:latest` to Docker Hub on manual trigger
(`workflow_dispatch`, deliberately not on every push — the image is
huge and most commits don't touch anything the image needs to reflect).

**Before it can run:**

1. Add two repo secrets (Docker Hub → Account Settings → Security → New
   Access Token, then `gh secret set DOCKERHUB_USERNAME` and
   `gh secret set DOCKERHUB_TOKEN`, or via the GitHub web UI).
2. GitHub-hosted runners have no NVIDIA GPU, so this workflow can only
   rebuild *this repo's* layer — it cannot build `isaac-lab-base`'s own
   image itself (that step needs the real Isaac Sim installer/GPU driver
   present, done once on this workstation via `container.py start base`).
   Push `isaac-lab-base:latest` to Docker Hub once from here first, then
   change `docker/Dockerfile`'s `FROM isaac-lab-base:latest` line to
   reference that pushed image instead of the bare local tag, before the
   workflow's own build step can succeed.

**Licensing consideration, worth resolving before making the Docker Hub
repo public:** the bundled AR4 description assets are MIT-licensed
(`annin_ar4_description`, redistribution-friendly with attribution,
already satisfied by this README/repo crediting the upstream project) —
this will need re-checking whenever a future task bundles a different
robot's assets under a different license. The base image itself is
`nvcr.io/nvidia/isaac-sim`, NVIDIA's own proprietary Omniverse/Isaac Sim
runtime under its own EULA — NVIDIA's container EULAs commonly restrict
redistributing the runtime to third parties. Recommend keeping the
Docker Hub repo **private** unless/until NVIDIA's exact terms have been
checked.
