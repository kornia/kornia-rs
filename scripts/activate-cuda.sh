#!/bin/bash
# Activation script for CUDA feature
# Ensures nvcc uses the conda-provided GCC (required for CUDA 12.x), on
# whatever architecture the env was solved for. Falls back to the system
# compiler when conda ships none for this arch (e.g. Jetson/aarch64 images
# where the cross-named binary does not exist).

_conda_gcc="${CONDA_PREFIX}/bin/$(uname -m)-conda-linux-gnu-gcc"
_conda_gxx="${CONDA_PREFIX}/bin/$(uname -m)-conda-linux-gnu-g++"

if [ -x "${_conda_gcc}" ]; then
    export CC="${_conda_gcc}"
    export CXX="${_conda_gxx}"
    export NVCC_CCBIN="${_conda_gcc}"
fi
unset _conda_gcc _conda_gxx
