#!/bin/bash
set -eo pipefail
export PATH=/mnt/data/venvs/cmake-tools/bin:/usr/local/cuda-12.6/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=/mnt/data/venvs/cmake-tools
export CPATH=/mnt/data/localdeps/root/usr/include
export PKG_CONFIG_PATH=/mnt/data/localdeps/root/usr/lib/aarch64-linux-gnu/pkgconfig:/mnt/data/localdeps/root/usr/share/pkgconfig
export CMAKE_PREFIX_PATH=/mnt/data/localdeps/root/usr
export CMAKE_INCLUDE_PATH=/mnt/data/localdeps/root/usr/include
export CMAKE_LIBRARY_PATH=/mnt/data/localdeps/root/usr/lib/aarch64-linux-gnu
export LIBRARY_PATH=/mnt/data/localdeps/root/usr/lib/aarch64-linux-gnu
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
B=/mnt/data/o3d-cuda-build
cmake -S /mnt/data/src/Open3D -B $B -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_MODULE=ON -DWITH_STUBGEN=OFF \
 -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_CUDA_COMPILER=$CUDACXX -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF \
 -DBUILD_PYTHON_MODULE=ON -DPython3_EXECUTABLE=/usr/bin/python3.10 -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_BENCHMARKS=OFF \
 -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF -DBUILD_LIBREALSENSE=OFF -DBUILD_JUPYTER_EXTENSION=OFF \
 -DUSE_BLAS=ON -DGLIBCXX_USE_CXX11_ABI=ON -DCMAKE_CUDA_FLAGS='--threads 1' -DBUILD_ISPC_MODULE=OFF
cmake --build $B -j2 -- pip-package
ls $B/lib/python_package/pip_package/
