#!/bin/bash
# One VDBFusion run. Usage: vdb_run.sh SCENE VOXEL NFRAMES TAG [extra bench args...]
set -u
SCENE=$1; V=$2; NF=$3; TAG=$4; shift 4
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
OUT=/mnt/data/kr-tsdf-spikes/results/baselines/vdbfusion/runs/${SCENE}_v${V}_n${NF}${TAG}
rm -rf "$OUT"; mkdir -p "$OUT"
for i in $(seq 1 180); do
  H=$($S/heavy_procs.sh); [ -z "$H" ] && break; echo "$(date +%T) waiting for: $H" >> $OUT/waited.txt; sleep 10
done
$S/clock_state.sh > $OUT/clock_start.txt 2>&1
CMD=(env LD_LIBRARY_PATH=/mnt/data/localdeps/root/usr/lib/aarch64-linux-gnu /usr/bin/python3.10 -I $S/vdb_bench.py --scene $SCENE --voxel $V --nframes $NF --out $OUT "$@")
echo "${CMD[@]}" > $OUT/command.txt
/mnt/data/venvs/tsdf-bench/bin/python -I $S/mem_sampler.py $OUT/mem.csv vdb_bench 0.05 &
SP=$!; sleep 0.7
T0=$(date +%s.%N)
/mnt/data/venvs/tsdf-bench/bin/python -I $S/timev.py $OUT/timev.json -- "${CMD[@]}" > $OUT/stdout.txt 2> $OUT/run.log
RC=$?; T1=$(date +%s.%N)
touch $OUT/done; wait $SP
$S/clock_state.sh > $OUT/clock_end.txt 2>&1
echo "rc=$RC wall_s=$(echo "$T1 - $T0" | bc)" > $OUT/exit.txt
cat $OUT/exit.txt $OUT/stdout.txt
