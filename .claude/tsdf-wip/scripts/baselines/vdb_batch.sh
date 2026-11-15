#!/bin/bash
# Sequential full runs (never two timed processes at once).
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
for cfg in "room0 0.05" "office0 0.05" "room0 0.02" "office0 0.02"; do
  set -- $cfg; echo "$(date +%T) start $1 $2"; $S/vdb_run.sh $1 $2 2000 ""; echo "$(date +%T) end $1 $2"
done
