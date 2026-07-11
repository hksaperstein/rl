#!/bin/bash
# Render all 6 shards for detection_v2_closeup
# This script runs shards in parallel with nohup to ensure they're fully detached

cd /home/saps/projects/rl/vision
rm -rf data/detection_v2_closeup
mkdir -p data/detection_v2_closeup

LOGDIR=/tmp/claude-1000/-home-saps-projects-rl/11bcb57b-1582-429a-be7f-4f19ff66965a/scratchpad

# Launch all 6 shards in parallel with nohup
for k in 0 1 2 3 4 5; do
  nohup /snap/bin/blender --background --python scripts/render_detection_dataset.py -- \
    --count 3000 --seed 5150 --shard $k --shards 6 --closeup --outdir data/detection_v2_closeup \
    > $LOGDIR/dv2_shard$k.log 2>&1 &
  echo "Shard $k PID: $!"
done

echo "All shards launched. Waiting for completion..."

# Wait for all background jobs to complete
wait

echo "All shards finished"
