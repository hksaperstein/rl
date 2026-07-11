#!/bin/bash
# Detached orchestrator for the full detection dataset generation.
cd /home/saps/projects/Dice-Detection
LOGDIR=/tmp/claude-1000/-home-saps-projects-Dice-Detection/402a5d46-22a8-48d9-a93d-13dd130b3d36/scratchpad
for k in 0 1 2 3 4 5; do
  blender --background --python scripts/render_detection_dataset.py -- \
    --count 10000 --seed 424242 --shard $k --shards 6 --outdir data/detection_v1 \
    > $LOGDIR/dv1d_shard$k.log 2>&1 &
done
wait
python3 scripts/render_detection_dataset.py --merge --outdir data/detection_v1 \
  > $LOGDIR/dv1d_merge.log 2>&1
touch data/detection_v1/GENERATION_COMPLETE
