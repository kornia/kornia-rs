#!/bin/bash
# Score every mesh.ply of finished 2000-frame runs (CPU; waits while any fuse_replica GPU run is going).
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
for d in /mnt/data/kr-tsdf-spikes/results/baselines/nvblox/runs/*_n2000; do
  [ -s $d/mesh.ply ] || continue; [ -s $d/score.json ] && continue
  case $(basename $d) in *_cuda_*) continue;; esac   # cuda-trace runs: same mesh as the nvtx/plain twin, not scored
  while pgrep -x fuse_replica > /dev/null; do sleep 10; done
  sc=$(basename $d | cut -d_ -f1)
  /mnt/data/venvs/tsdf-bench/bin/python -I $S/score_mesh.py $d/mesh.ply --gt /mnt/data/datasets/replica/Replica/${sc}_mesh.ply \
     --cull-to-frustum --traj /mnt/data/datasets/replica/Replica/$sc/traj.txt --frames 0:2000:1 --label nvblox_$(basename $d) > $d/score.json 2> $d/score.err
  echo "$(basename $d) $(head -c 100 $d/score.json)"
done
