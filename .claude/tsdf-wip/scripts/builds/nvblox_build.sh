#!/bin/bash
set -e
export PATH=/mnt/data/venvs/cmake-tools/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/mnt/data/venvs/cmake-tools
export CUDACXX=/usr/local/cuda/bin/nvcc
B=/mnt/data/nvblox-build
cmake -S /mnt/data/src/nvblox-v0.0.10 -B $B -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF \
  -DBUILD_EXPERIMENTS=OFF -DBUILD_PYTORCH_WRAPPER=OFF -DBUILD_RENDERER=OFF
cmake --build $B --target fuse_replica -j1
