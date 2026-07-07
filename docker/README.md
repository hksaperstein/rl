# Docker: local development only, not for redistribution

Packages this repo's code + whatever pre-built USD assets the current
task needs into a Docker image, built on Isaac Lab's own official base
image, so `scripts/train.py` runs unmodified inside a container. Generic
to this repo as a whole (a reusable Isaac Lab manipulation research
platform, per CLAUDE.md's North Star), not hardcoded to any one
robot/task — today that's AR4 pick-and-place, but nothing here assumes
that stays true.

## Important: do not push this image (or `isaac-lab-base`) to any external registry

**The NVIDIA Isaac Sim Additional Software and Materials License
explicitly prohibits redistribution.** Verified directly from the
license text (nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/):

> "You may not... sell, rent, sublicense, transfer, distribute or
> otherwise make available to others... any portion of the Software"

— with authorized users limited to employees/contractors of your own
entity, accessing from your own secure network. This applies regardless
of whether a destination registry repo is public or private; a private
Docker Hub/ECR/Artifact Registry repo is still "making available to
others" in the license's sense once anyone outside your own org/network
could pull it (and Docker Hub, ECR, etc. are third-party infrastructure,
not "your secure network").

**Practical consequence:** any image built `FROM isaac-lab-base` (which
is itself `FROM nvcr.io/nvidia/isaac-sim`) must stay local to the
machine it's built on, or move only within your own private
infrastructure — never pushed to Docker Hub, ECR, GCP Artifact Registry,
or any other third-party-operated registry, public or private tier.
This repo's own Docker Hub repo (`hsaperstein/rl`) must not receive a
tag containing the Isaac Sim runtime.

## What Docker is actually useful for here

Local development/testing on this workstation (or any single machine
you personally control, under your own NGC/Isaac Sim EULA acceptance) —
building the image, running it with `--gpus all` once
`nvidia-container-toolkit` is installed, iterating without touching the
host Isaac Lab install. Verified working end-to-end (build succeeds, a
real 24GB image builds cleanly).

## Cloud GPU deployment: build fresh on each instance, don't redistribute a baked image

The EULA-compliant pattern (and what NVIDIA's own tooling does) is:
each cloud instance pulls Isaac Sim **directly from NVIDIA's NGC
registry** and builds locally there, under that instance's own EULA
acceptance — never receiving a pre-baked copy from an intermediary
third-party registry.

**Recommended path: NVIDIA's own [IsaacAutomator](https://github.com/isaac-sim/IsaacAutomator)**
(`./build` then `./deploy-<gcp|aws|azure|alibaba>`) — deploys a fully
configured Isaac Sim/Isaac Lab workstation to a cloud GPU instance,
pulling Isaac Sim from NVIDIA directly on that instance, no
redistribution involved. Then `git clone` (or `rsync`) this repo's code
onto that instance and run `scripts/train.py` there directly — this
repo's own code has no licensing restriction (it's the user's own
work), only the Isaac Sim runtime itself is restricted.

**If building this repo's own Dockerfile on a cloud instance instead**
(e.g. inside a RunPod/Lambda pod that doesn't use IsaacAutomator): run
`docker/Dockerfile`'s two build steps *on that instance itself* — same
commands as the local build below, just executed there instead of here
— rather than building locally and shipping the result. Each instance
independently pulls `nvcr.io/nvidia/isaac-sim` from NVIDIA and accepts
the EULA itself.

## Prerequisites (this workstation, before building)

1. `assets/` must already contain whatever the current task needs (run
   that task's own asset-build script first, e.g. `scripts/build_asset.py`
   for the AR4 work) — the Dockerfile fails the build early with a clear
   error if `assets/` is empty. Already satisfied as of this writing.
2. Docker is installed locally (confirmed: 27.3.1). `nvidia-container-toolkit`
   is **not** currently installed on this workstation, which means the
   image can be *built* here but not *run with GPU access* here — local
   test-running would need `sudo apt install nvidia-container-toolkit`
   first if ever wanted (a real system change, not done as part of this
   work).

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
docker build -f docker/Dockerfile -t rl:latest .
```

## Run (local, with GPU passthrough - needs `nvidia-container-toolkit` first)

```bash
docker run --rm -it --gpus all --network host \
    rl:latest \
    ./isaaclab.sh -p scripts/train.py --pregrasp --num_envs 4096 --headless
```

## Per-provider cloud account setup (manual — needs your own payment method/identity)

- **GCP**: console.cloud.google.com → create project → enable Compute
  Engine API → request GPU quota (often starts at 0) — IsaacAutomator
  supports this directly.
- **Lambda Labs**: cloud.lambdalabs.com → sign up → add payment method →
  on-demand instances available immediately, no quota request typically
  needed. (Not IsaacAutomator-supported — build the image manually on
  the instance per the "cloud GPU deployment" section above.)
- **RunPod**: runpod.io → sign up → add payment method → Community Cloud
  pods available immediately. (Same manual-build note as Lambda.)

AWS deliberately out of scope (SageMaker's managed-training premium buys
nothing this repo's single-researcher workflow needs, and plain EC2 was
dropped too per direct instruction).

Once you have credentials/API keys for whichever provider(s) you want,
hand them to me (or set them as environment variables / CLI-configured
credentials in this environment) and I can drive the actual instance
launch and job dispatch from here.
