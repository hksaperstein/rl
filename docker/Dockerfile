# Cloud/headless training image for this repo's AR4 pick-and-place work.
#
# Builds on Isaac Lab's own official base image (isaac-lab-base) rather
# than reimplementing its install steps - that image already has Isaac
# Sim 5.1.0 + Isaac Lab v2.3.1 (matching this workstation's installed
# versions) fully set up. This Dockerfile only adds this repo's own
# code and the pre-built AR4 USD asset on top.
#
# Build (two steps, from this repo's root):
#
#   1. Build the isaac-lab-base image (one-time, or whenever Isaac Lab
#      version changes) using Isaac Lab's own official docker-compose:
#
#        cd /home/saps/IsaacLab/docker
#        echo "ACCEPT_EULA=Y" > .env.base.local  # already in .env.base by default
#        docker compose --profile base build
#
#      This produces a local image tagged `isaac-lab-base:latest`.
#
#   2. Build this repo's image on top of it:
#
#        cd /home/saps/projects/rl
#        docker build -f docker/Dockerfile -t ar4-pickplace:latest .
#
# Run (locally, with GPU passthrough - requires nvidia-container-toolkit
# installed on the host, which this workstation does not currently have):
#
#   docker run --rm -it --gpus all --network host \
#       ar4-pickplace:latest \
#       ./isaaclab.sh -p scripts/train.py --pregrasp --num_envs 4096 --headless
#
# On a cloud GPU instance (AWS/GCP/RunPod/Lambda - all support
# --gpus all / nvidia container runtime out of the box on their GPU
# instance images), push this image to a registry the instance can pull
# from (Docker Hub, AWS ECR, GCP Artifact Registry) and run the same
# command there. See docker/README.md for the full per-provider walkthrough.

FROM isaac-lab-base:latest

ENV DEBIAN_FRONTEND=noninteractive

# This repo's own code - tasks/, scripts/, docs/, top-level configs.
# Excludes logs/, .git/, kb/.obsidian/, etc. via docker/.dockerignore.
# Deliberately does NOT exclude assets/ (gitignored in this repo, but
# Docker's build context is governed by .dockerignore, not .gitignore) -
# the pre-built AR4 USD is bundled directly, so the cloud instance never
# needs the external ROS2/xacro toolchain (annin_ar4_description) this
# repo's scripts/build_asset.py depends on to regenerate it from source.
WORKDIR /workspace/ar4-pickplace
COPY . /workspace/ar4-pickplace

# Fail fast at build time if the asset wasn't built locally before
# `docker build` was run - a container without it can't run anything.
RUN test -f /workspace/ar4-pickplace/assets/ar4_mk5/ar4_mk5.usd || \
    (echo "ERROR: assets/ar4_mk5/ar4_mk5.usd missing from build context." \
          "Run scripts/build_asset.py locally before building this image." \
          "See docker/README.md." && exit 1)

# scripts/hillclimb_rewards.py hardcodes ISAACLAB_SH = "/home/saps/IsaacLab/isaaclab.sh"
# as a real runtime constant (confirmed by grepping this repo's scripts/
# for non-docstring /home/saps references - every other hit is a
# docstring usage example, harmless inside the container). Symlink so
# that one script also works unmodified here; inside the container Isaac
# Lab actually lives at /workspace/isaaclab (Isaac Lab's own
# DOCKER_ISAACLAB_PATH).
RUN mkdir -p /home/saps && ln -sf /workspace/isaaclab /home/saps/IsaacLab

ENTRYPOINT ["/bin/bash"]
