#!/bin/bash
# Untimed sizing passes (CUDA, block_count 1000, hashmap auto-grows): required capacity per config.
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
for SC in room0 office0; do for V in 0.05 0.02; do for ROW in M R; do
  /mnt/data/venvs/tsdf-bench/bin/python -I $S/evict_page_cache.py /mnt/data/datasets/replica/Replica/$SC > /dev/null
  $S/o3d_run.sh $SC $V $ROW cuda size 1000 plain 2000
done; done; done
