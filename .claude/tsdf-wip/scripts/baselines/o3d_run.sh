#!/bin/bash
# One Open3D VBG run. Usage: o3d_run.sh SCENE VOXEL ROW(M|R) DEV(cuda|cpu) MODE(size|time) BLOCK_COUNT PROF(plain|nsys) [NFRAMES] [TAG]
set -u
SCENE=$1; V=$2; ROW=$3; DEV=$4; MODE=$5; BC=$6; PROF=$7; NF=${8:-2000}; TAG=${9:-}
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
OUT=/mnt/data/kr-tsdf-spikes/results/baselines/open3d/runs/${SCENE}_v${V}_${ROW}_${DEV}_${MODE}_${PROF}_n${NF}${TAG}
rm -rf "$OUT"; mkdir -p "$OUT"
if [ "$DEV" = cuda ]; then
  PY=(env PYTHONPATH=/mnt/data/venvs/o3d-cuda-site:/mnt/data/venvs/o3d-cuda-deps /usr/bin/python3.10); D=CUDA:0
else
  PY=(/mnt/data/venvs/tsdf-bench/bin/python -I); D=CPU:0
fi
PRE=(); EXTRA=()
[ "$PROF" = nsys ] && PRE=(/usr/local/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-memory-usage=true --export=sqlite -f true -o $OUT/prof) && EXTRA=(--nvtx)
# hygiene: wait (max 30 min) while another heavy job runs
for i in $(seq 1 180); do
  H=$($S/heavy_procs.sh)
  [ -z "$H" ] && break; echo "$(date +%T) waiting for: $H" >> $OUT/waited.txt; sleep 10
done
$S/clock_state.sh > $OUT/clock_start.txt 2>&1
CMD=("${PRE[@]}" "${PY[@]}" $S/o3d_vbg_bench.py --scene $SCENE --voxel $V --row $ROW --device $D --block-count $BC --nframes $NF --mode $MODE --out $OUT "${EXTRA[@]}")
echo "${CMD[@]}" > $OUT/command.txt
/mnt/data/venvs/tsdf-bench/bin/python -I $S/mem_sampler.py $OUT/mem.csv o3d_bench 0.05 &
SP=$!
sleep 0.7
T0=$(date +%s.%N)
/mnt/data/venvs/tsdf-bench/bin/python -I $S/timev.py $OUT/timev.json -- "${CMD[@]}" > $OUT/stdout.txt 2> $OUT/run.log
RC=$?
T1=$(date +%s.%N)
touch $OUT/done; wait $SP
$S/clock_state.sh > $OUT/clock_end.txt 2>&1
echo "rc=$RC wall_s=$(echo "$T1 - $T0" | bc)" > $OUT/exit.txt
cat $OUT/exit.txt $OUT/stdout.txt
