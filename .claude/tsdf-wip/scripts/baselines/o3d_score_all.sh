#!/bin/bash
# Score every timed plain run mesh with the shared scorer (CPU, untimed; run only when no timed job is running).
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts; R=/mnt/data/datasets/replica/Replica
for d in /mnt/data/kr-tsdf-spikes/results/baselines/open3d/runs/*_time_plain_n2000; do
  [ -f $d/score.json ] && continue
  SC=$(basename $d | cut -d_ -f1)
  if [ ! -f $d/mesh.ply ]; then echo "{\"empty\": true, \"note\": \"no mesh.ply written (0 vertices)\"}" > $d/score.json; continue; fi
  CMD="/mnt/data/venvs/tsdf-bench/bin/python -I $S/score_mesh.py $d/mesh.ply --gt $R/${SC}_mesh.ply --cull-to-frustum --traj $R/$SC/traj.txt --frames 0:2000:1"
  echo "$CMD" > $d/score_cmd.txt
  $CMD > $d/score.json 2> $d/score.log; echo "$(basename $d) rc=$?"
done
