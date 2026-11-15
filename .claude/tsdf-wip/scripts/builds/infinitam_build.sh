#!/bin/bash
# InfiniTAM v3 (origin/infinitam_v3 c423b89b) CUDA build, headless (InfiniTAM_cli only). Re-runnable.
# Usage: infinitam_build.sh [s|rgb]
#   s   (default): ITMVoxel_s (library default typedef, no colour) in /mnt/data/src/InfiniTAM -> /mnt/data/infinitam-build   (Row R)
#   rgb          : ITMVoxel_s_rgb (colour) in a separate source export /mnt/data/src/InfiniTAM-rgb -> /mnt/data/infinitam-build-rgb (Row M)
# Both trees carry ONLY the bench patch /mnt/data/kr-tsdf-spikes/infinitam_patch/bench.patch (CLI-only: env overrides of
# voxel/mu/maxW/frustum/trackerConfig, settings echo, NVTX ranges, block count, timed MeshScene, PLY writer); ITMLib untouched
# except the one-line ITMVoxel typedef in the rgb tree.
set -eo pipefail
V=${1:-s}
export PATH=/mnt/data/venvs/cmake-tools/bin:/usr/local/cuda-12.6/bin:$PATH
export PYTHONPATH=/mnt/data/venvs/cmake-tools
P=/mnt/data/kr-tsdf-spikes/infinitam_patch/bench.patch
if [ "$V" = s ]; then
  S=/mnt/data/src/InfiniTAM; B=/mnt/data/infinitam-build
  git -C $S apply --reverse --check $P 2>/dev/null || git -C $S apply $P
else
  S=/mnt/data/src/InfiniTAM-rgb; B=/mnt/data/infinitam-build-rgb
  if [ ! -d $S ]; then
    mkdir -p $S; git -C /mnt/data/src/InfiniTAM archive c423b89b | tar -x -C $S
    (cd $S && patch -p1 < $P)
    sed -i 's/^typedef ITMVoxel_s ITMVoxel;/typedef ITMVoxel_s_rgb ITMVoxel;/' $S/InfiniTAM/ITMLib/ITMLibDefines.h
  fi
  grep -q '^typedef ITMVoxel_s_rgb ITMVoxel;' $S/InfiniTAM/ITMLib/ITMLibDefines.h
fi
cmake -S $S/InfiniTAM -B $B -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 -DCUDA_COMPUTE_CAPABILITY=87 \
  -DWITH_OPENMP=OFF -DWITH_PNG=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Wno-dev \
  -DGLUT_LIBRARY=/usr/lib/aarch64-linux-gnu/libm.so -DGLUT_INCLUDE_DIR=/usr/include \
  -DCUDA_NVCC_FLAGS="--threads;1"
cmake --build $B -j1 --target InfiniTAM_cli
ls -l $B/Apps/InfiniTAM_cli/InfiniTAM_cli
