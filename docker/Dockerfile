# Cloud/headless training image for this repo (a general, reusable
# Isaac Lab manipulation research platform - see CLAUDE.md's North Star;
# not permanently scoped to any one arm/task, even though AR4
# pick-and-place is the only work in progress today).
#
# Builds on Isaac Lab's own official base image (isaac-lab-base) rather
# than reimplementing its install steps - that image already has Isaac
# Sim 5.1.0 + Isaac Lab v2.3.1 (matching this workstation's installed
# versions) fully set up. This Dockerfile only adds this repo's own
# code and whatever pre-built USD assets the current task needs on top.
#
# Build (two steps, from this repo's root):
#
#   1. Build the isaac-lab-base image (one-time, or whenever Isaac Lab
#      version changes) using Isaac Lab's own official wrapper - see
#      docker/README.md for the exact command (raw `docker compose
#      --profile base build` does not work directly, env var
#      interpolation gotcha, documented there).
#
#   2. Build this repo's image on top of it:
#
#        cd /home/saps/projects/rl
#        docker build -f docker/Dockerfile -t isaaclab-manipulation-rl:latest .
#
# Run (locally, with GPU passthrough - requires nvidia-container-toolkit
# installed on the host, which this workstation does not currently have):
#
#   docker run --rm -it --gpus all --network host \
#       isaaclab-manipulation-rl:latest \
#       ./isaaclab.sh -p scripts/train.py --pregrasp --num_envs 4096 --headless
#
# On a cloud GPU instance (AWS/GCP/RunPod/Lambda - all support
# --gpus all / nvidia container runtime out of the box on their GPU
# instance images), push this image to a registry the instance can pull
# from (Docker Hub, AWS ECR, GCP Artifact Registry) and run the same
# command there. See docker/README.md for the full per-provider walkthrough.
#
# BASE_IMAGE defaults to the local tag (fast, no pull, for building on
# this workstation where isaac-lab-base already exists locally) - CI
# (GitHub-hosted runners have no GPU to build isaac-lab-base themselves)
# overrides it to the pushed Docker Hub reference:
# --build-arg BASE_IMAGE=hsaperstein/rl:isaac-lab-base

ARG BASE_IMAGE=isaac-lab-base:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# This repo's own code - tasks/, scripts/, docs/, top-level configs, for
# whichever robot/task combination is currently active (not just AR4 -
# this Dockerfile doesn't know or care which). Excludes logs/, .git/,
# kb/.obsidian/, etc. via docker/.dockerignore. Deliberately does NOT
# exclude assets/ (gitignored in this repo, but Docker's build context
# is governed by .dockerignore, not .gitignore) - whatever's already
# been built locally via that task's own build_asset.py-equivalent gets
# bundled directly, so the cloud instance never needs whatever external
# toolchain (e.g. AR4's ROS2/xacro annin_ar4_description dependency)
# produced it.
WORKDIR /workspace/rl
COPY . /workspace/rl

# Fail fast at build time if no assets were built locally before
# `docker build` was run - a container with an empty assets/ directory
# can't run anything, regardless of which task it's for.
RUN [ -n "$(find /workspace/rl/assets -mindepth 1 -print -quit 2>/dev/null)" ] || \
    (echo "ERROR: assets/ is empty or missing from the build context." \
          "Run this task's own asset-build script locally first" \
          "(e.g. scripts/build_asset.py for the AR4 work)." \
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
