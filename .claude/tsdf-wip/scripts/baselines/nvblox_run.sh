#!/bin/bash
# One nvblox v0.0.10 fuse_replica run. Usage: nvblox_run.sh SCENE VOXEL ROW(M|R) MODE(inc|full|nomesh) PROF(plain|nvtx|cuda) [NFRAMES]
# inc: mesh every frame (nvblox incremental); full: no per-frame mesh, one full mesh after the loop + map; nomesh: no color mesh at all + map
set -u
SCENE=$1; V=$2; ROW=$3; MODE=$4; PROF=$5; NF=${6:-2000}
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
BIN=/mnt/data/nvblox-build/executables/fuse_replica
DATA=/mnt/data/datasets/replica/Replica/$SCENE
OUT=/mnt/data/kr-tsdf-spikes/results/baselines/nvblox/runs/${SCENE}_v${V}_${ROW}_${MODE}_${PROF}_n${NF}
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(--voxel_size=$V --num_frames=$NF --esdf_frame_subsampling=-1 --visualization=false --timing_output_path=$OUT/timing.txt)
[ "$MODE" != nomesh ] && FLAGS+=(--mesh_output_path=$OUT/mesh.ply)
[ "$MODE" = nomesh ] && FLAGS+=(--mesh_frame_subsampling=-1 --map_output_path=$OUT/map.nvblx)
[ "$MODE" = inc ] && FLAGS+=(--mesh_frame_subsampling=1)
[ "$MODE" = full ] && FLAGS+=(--mesh_frame_subsampling=-1 --map_output_path=$OUT/map.nvblx)
[ "$ROW" = M ] && FLAGS+=(--projective_integrator_weighting_mode=0 --projective_integrator_max_weight=64
       --projective_integrator_truncation_distance_vox=4 --projective_integrator_max_integration_distance_m=7
       --mesh_integrator_min_weight=1)
PRE=()
[ "$PROF" = nvtx ] && PRE=(/usr/local/bin/nsys profile -t nvtx -s none --cpuctxsw=none --export=sqlite -f true -o $OUT/prof)
[ "$PROF" = cuda ] && PRE=(/usr/local/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-memory-usage=true --export=sqlite -f true -o $OUT/prof)
# hygiene: wait (max 30 min) while another heavy job runs
for i in $(seq 1 180); do
  H=$(pgrep -a -f 'cargo|cc1plus|rustc|nvcc|ninja|fuse_replica|nsys|score_mesh|o3d|open3d' | grep -v -e nvblox_run -e pgrep)
  [ -z "$H" ] && break; echo "waiting for: $H" >> $OUT/waited.txt; sleep 10
done
$S/clock_state.sh > $OUT/clock_start.txt 2>&1
echo "${PRE[@]} $BIN $DATA ${FLAGS[@]}" > $OUT/command.txt
/mnt/data/venvs/tsdf-bench/bin/python -I $S/mem_sampler.py $OUT/mem.csv fuse_replica 0.05 &
SP=$!
sleep 0.7
T0=$(date +%s.%N)
/mnt/data/venvs/tsdf-bench/bin/python -I $S/timev.py $OUT/timev.json -- "${PRE[@]}" $BIN $DATA "${FLAGS[@]}" > $OUT/stdout.txt 2> $OUT/run.log
RC=$?
T1=$(date +%s.%N)
touch $OUT/done; wait $SP
$S/clock_state.sh > $OUT/clock_end.txt 2>&1
echo "rc=$RC wall_s=$(echo "$T1 - $T0" | bc)" > $OUT/exit.txt
rm -f $OUT/map.nvblx.tmp
cat $OUT/exit.txt
