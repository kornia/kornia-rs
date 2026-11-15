#!/bin/bash
# Sequential queue: one GPU run at a time, parse (CPU, seconds) between runs. Usage: nvblox_batch.sh VOXEL "SCENES" "ROWS" [EXTRA_SPECS...]
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
RUNS=/mnt/data/kr-tsdf-spikes/results/baselines/nvblox/runs
V=$1; SCENES=$2; ROWS=$3; shift 3
specs=()
for sc in $SCENES; do for r in $ROWS; do
  specs+=("$sc $V $r inc nvtx 2000" "$sc $V $r full plain 2000" "$sc $V $r inc cuda 2000"); done; done
specs+=("$@")
for sp in "${specs[@]}"; do
  set -- $sp
  echo "$(date +%T) START $sp"
  $S/nvblox_run.sh $sp
  d=$RUNS/${1}_v${2}_${3}_${4}_${5}_n${6}
  rm -f $d/prof.nsys-rep.tmp
  /mnt/data/venvs/tsdf-bench/bin/python -I $S/nvblox_parse.py $d > $d/parse.out 2>&1
  echo "$(date +%T) END $sp $(cat $d/exit.txt)"
done
