#!/bin/bash
# One InfiniTAM v3 CUDA run at GT poses (file tracker). Usage:
#   infinitam_run.sh SCENE VOXEL ROW(M|R) PROF(plain|nsys) REP [NFRAMES=2000] [MESH_MAXTRI=default] [TAG]
# Row M -> /mnt/data/infinitam-build-rgb (ITMVoxel_s_rgb, colour), mu 4 vox, maxW 64, frustum [0.1, 7.0]
# Row R -> /mnt/data/infinitam-build     (ITMVoxel_s, library default, no colour), mu 4 vox, maxW 100 + frustum 0.2/3.0 (ITMLibSettings defaults)
set -u
SCENE=$1; V=$2; ROW=$3; PROF=$4; REP=$5; NF=${6:-2000}; MAXTRI=${7:-default}; TAG=${8:-}
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
D=/mnt/data/datasets/replica/infinitam
OUT=/mnt/data/kr-tsdf-spikes/results/baselines/infinitam/runs/${SCENE}_v${V}_${ROW}_${PROF}_n${NF}_r${REP}${TAG}
rm -rf "$OUT"; mkdir -p "$OUT"
IN=$D/$SCENE
if [ "$NF" != 2000 ]; then
  IN=$D/${SCENE}_n$NF; mkdir -p $IN; ln -sf $D/$SCENE/calib.txt $IN/calib.txt
  for i in $(seq -f %04g 0 $((NF-1))); do ln -sf $D/$SCENE/$i.ppm $IN/$i.ppm; ln -sf $D/$SCENE/$i.pgm $IN/$i.pgm; done
fi
ENVV=(INFINITAM_VOXEL=$V INFINITAM_MU_VOX=4 "INFINITAM_TRACKER=type=file,mask=$D/$SCENE/poses/%04i.txt" INFINITAM_MESH_OUT=$OUT/mesh.ply INFINITAM_MESH_WARM=3)
[ "$MAXTRI" != default ] && ENVV+=(INFINITAM_MESH_MAXTRI=$MAXTRI)
if [ "$ROW" = M ]; then BIN=/mnt/data/infinitam-build-rgb/Apps/InfiniTAM_cli/InfiniTAM_cli; ENVV+=(INFINITAM_MAXW=64 INFINITAM_ZMIN=0.1 INFINITAM_ZMAX=7.0)
else BIN=/mnt/data/infinitam-build/Apps/InfiniTAM_cli/InfiniTAM_cli; fi
PRE=()
[ "$PROF" = nsys ] && PRE=(/usr/local/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-memory-usage=true --export=sqlite -f true -o $OUT/prof)
# hygiene: wait (max 30 min) while another heavy job runs (incl. another InfiniTAM_cli or the converter)
for i in $(seq 1 180); do
  H="$($S/heavy_procs.sh)$(pgrep -a -x InfiniTAM_cli)$(pgrep -a -f replica_to_infinitam | grep -v pgrep)"
  [ -z "$H" ] && break; echo "$(date +%T) waiting for: $H" >> $OUT/waited.txt; sleep 10
done
$S/clock_state.sh > $OUT/clock_start.txt 2>&1
CMD=(env "${ENVV[@]}" "${PRE[@]}" $BIN $IN/calib.txt "$IN/%04i.ppm" "$IN/%04i.pgm")
printf '%q ' "${CMD[@]}" > $OUT/command.txt; echo >> $OUT/command.txt
/mnt/data/venvs/tsdf-bench/bin/python -I $S/mem_sampler.py $OUT/mem.csv InfiniTAM_cli 0.05 &
SP=$!
sleep 0.7
T0=$(date +%s.%N)
/mnt/data/venvs/tsdf-bench/bin/python -I $S/timev.py $OUT/timev.json -- "${CMD[@]}" > $OUT/stdout.txt 2> $OUT/run.log
RC=$?
T1=$(date +%s.%N)
touch $OUT/done; wait $SP
$S/clock_state.sh > $OUT/clock_end.txt 2>&1
echo "rc=$RC wall_s=$(echo "$T1 - $T0" | bc)" > $OUT/exit.txt
cat $OUT/exit.txt; grep -v "^frame" $OUT/stdout.txt | grep -v "^using\|^initialis"; tail -2 $OUT/run.log
