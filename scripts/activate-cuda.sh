#!/bin/bash
# Activation script for CUDA feature
# Ensures nvcc uses the conda-provided GCC 12 (required for CUDA 12.x)

# Use the actual binary (not the potentially broken symlink)
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export NVCC_CCBIN="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
