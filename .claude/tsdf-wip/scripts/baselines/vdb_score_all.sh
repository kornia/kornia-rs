#!/bin/bash
# Score every full VDBFusion mesh (untimed; only after the timed batch exits).
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts; R=/mnt/data/datasets/replica/Replica
for d in /mnt/data/kr-tsdf-spikes/results/baselines/vdbfusion/runs/*_n2000; do
  SC=$(basename $d | cut -d_ -f1)
  /usr/bin/python3.10 -I $S/vdb_meta.py $d/grids.vdb > $d/vdb_meta.json
  for ROW in M R; do
    [ -f $d/score_$ROW.json ] && continue
    CMD="/mnt/data/venvs/tsdf-bench/bin/python -I $S/score_mesh.py $d/mesh_$ROW.ply --gt $R/${SC}_mesh.ply --cull-to-frustum --traj $R/$SC/traj.txt --frames 0:2000:1"
    echo "$CMD" > $d/score_${ROW}_cmd.txt
    $CMD > $d/score_$ROW.json 2> $d/score_$ROW.log; echo "$(basename $d) $ROW rc=$?"
  done
done
