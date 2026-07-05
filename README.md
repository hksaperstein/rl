# rl - AR4 Pick-and-Place RL

Everything here runs through Isaac Lab's launcher, not plain `python`. `isaaclab.sh`
lives in the separate IsaacLab install, not this repo - always run from this
repo's root and reference it by absolute path:

```bash
cd ~/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py [args]
```

## 1. Build the robot/scene assets (one-time)

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/build_asset.py
```

## 2. Sanity-check perception before trusting it anywhere else

Slides the cube across the camera's view for a few seconds and writes a labeled
mp4 - watch it before running anything else that depends on perception:

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/perception_calibration.py --headless
```

Check `logs/videos/perception_calibration.mp4`: the sliding cube should be
labeled `"cube"` throughout, and the three static objects (sphere, rectangular
prism, wedge) should be labeled correctly and consistently, without flickering
between shapes frame to frame.

**Known limitation:** The shape classifier currently misclassifies the real cube and rectangular prism as "sphere" in the calibration clip, with only the wedge classifying correctly. This is a threshold-tuning issue where parameters optimized on synthetic test data don't generalize to real sensor noise; it's tracked as a follow-up improvement and does not affect core pick-and-place functionality.

The perception math itself (ground-plane removal, shape classification,
tracking) has its own fast unit test suite, independent of Isaac Sim:

```bash
python3 -m pytest perception/tests/ -v
```

## 3. Train

```bash
# Quick smoke test first (~seconds, confirms the loop runs and writes a checkpoint):
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless

# Full training run:
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
```

Checkpoints and TensorBoard logs are written to `logs/train/<timestamp>/`.
Watch training with:

```bash
tensorboard --logdir logs/train
```

What to look at:

- `Train/mean_reward` - overall trend; should climb and plateau.
- `Episode_Reward/lifting_cube` - climbing off zero means the policy is at
  least starting to lift the cube, independent of whether it's placing
  accurately yet.
- `Episode_Reward/cube_goal_tracking_fine_grained` - the sharpest signal that
  placement is getting precise, not just "close enough."
- `Episode_Termination/cube_reached_goal` - the success rate: fraction of
  episodes that ended by actually reaching the goal, rather than timing out.
  This is the clearest single "is it working" number - reward can climb from
  partial credit (reaching, lifting) while this stays at zero, which tells you
  the policy is exploring but not yet succeeding.
- `Episode_Termination/time_out` - the complement of the above (episodes that
  ran out the clock without success).

There's no fixed "enough training" iteration count - stop once
`Episode_Termination/cube_reached_goal` has climbed and plateaued, rather than
running to a predetermined number of iterations.

## 4. Evaluate a checkpoint

```bash
# Privileged simulation state (fast, matches how training worked):
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<run>/model_<iter>.pt --episodes 10

# Real camera-based perception instead:
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<run>/model_<iter>.pt --episodes 10 --perception
```

Videos are written to `logs/videos/` (`ar4_pickplace-*.mp4` for the default
path, `ar4_pickplace_perception.mp4` for `--perception`, with the detection
overlay burned in for the latter). A healthy result: the cube is reliably
picked up and placed near the target region in most episodes.

## 5. Interactive demo

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/interactive_demo.py --checkpoint logs/train/<run>/model_<iter>.pt
```

With the GUI open: drag the cube anywhere in the workspace using the
viewport's drag gizmo, then let go. Once it's settled for about a second, the
arm picks it up and moves it to the target region on the other side, using the
real perception pipeline the whole time (not privileged simulation state) -
including through the brief period where the arm itself blocks the camera's
view of the cube mid-grasp (the tracker holds its last-known position through
that). Drag the cube outside the workspace or camera's view and the arm stays
idle rather than reacting to it.

The session records to `logs/videos/ar4_interactive_demo.mp4` with the
detection overlay burned in, and keeps running (watching for the next drag)
until you close the window.
