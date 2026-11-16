#!/bin/bash
# Timed matrix. Per config: CUDA plain (headline timing), CUDA nsys (device memory + launch proof, timing secondary), CPU plain.
# block_count = ceil(1.1 * required / 500) * 500, required = max_i(size_before_i + n_frustum_i) from the sizing pass.
S=/mnt/data/kr-tsdf-spikes/results/baselines/scripts
R=/mnt/data/kr-tsdf-spikes/results/baselines/open3d/runs
CONFIGS=${CONFIGS:-"room0:0.05:M room0:0.05:R room0:0.02:M room0:0.02:R office0:0.05:M office0:0.05:R office0:0.02:M office0:0.02:R"}
for C in $CONFIGS; do
  IFS=: read SC V ROW <<< "$C"
  REQ=$(/mnt/data/venvs/tsdf-bench/bin/python -I -c "import json;print(json.load(open('$R/${SC}_v${V}_${ROW}_cuda_size_plain_n2000/summary.json'))['required_capacity_max_size_plus_frustum'])")
  BC=$(( ( (REQ * 11 + 9) / 10 + 499) / 500 * 500 ))
  echo "$(date +%T) $SC $V $ROW required=$REQ block_count=$BC"
  for RUN in "cuda plain" "cuda nsys" "cpu plain"; do
    set -- $RUN
    /mnt/data/venvs/tsdf-bench/bin/python -I $S/evict_page_cache.py /mnt/data/datasets/replica/Replica/$SC > /dev/null
    $S/o3d_run.sh $SC $V $ROW $1 time $BC $2 2000
  done
done
echo "$(date +%T) batch done"
