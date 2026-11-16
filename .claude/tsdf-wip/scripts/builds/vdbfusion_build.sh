#!/bin/bash
# Builds OpenVDB (nachovizzo/openvdb nacho/vdbfusion fork, 9.1.1-based) -> VDBFusion C++ lib + pybind module. Re-runnable.
set -eo pipefail
LD=/mnt/data/localdeps/root/usr
export PATH=/mnt/data/venvs/cmake-tools/bin:$PATH
export PYTHONPATH=/mnt/data/venvs/cmake-tools
export CPATH=$LD/include
export LIBRARY_PATH=$LD/lib/aarch64-linux-gnu
export LD_LIBRARY_PATH=$LD/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$LD:/mnt/data/vdbfusion-build/install:/mnt/data/venvs/vdbfusion-buildtools/pybind11/share/cmake/pybind11
export PKG_CONFIG_PATH=$LD/lib/aarch64-linux-gnu/pkgconfig
PFX=/mnt/data/vdbfusion-build/install
# 1) OpenVDB (static core, no tools/python/AX/nanovdb, zlib off like upstream recipe)
cmake -S /mnt/data/src/openvdb-vdbfusion -B /mnt/data/vdbfusion-build/openvdb -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PFX -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DOPENVDB_BUILD_PYTHON_MODULE=OFF -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_BUILD_UNITTESTS=OFF \
  -DOPENVDB_BUILD_DOCS=OFF -DOPENVDB_BUILD_AX=OFF -DOPENVDB_BUILD_NANOVDB=OFF -DUSE_NANOVDB=OFF \
  -DOPENVDB_CORE_SHARED=OFF -DOPENVDB_CORE_STATIC=ON -DOPENVDB_CXX_STRICT=OFF -DUSE_ZLIB=OFF -DUSE_CCACHE=OFF \
  -DBLOSC_ROOT=$LD -DTBB_ROOT=$LD -DBOOST_ROOT=/usr -DUSE_PKGCONFIG=OFF -DCMAKE_CXX_FLAGS="-w"
cmake --build /mnt/data/vdbfusion-build/openvdb -j1 --target install
# 2) VDBFusion python (pip installs into target dir)
cd /mnt/data/src/vdbfusion
mkdir -p /mnt/data/venvs/vdbfusion
CMAKE_BUILD_PARALLEL_LEVEL=1 CMAKE_ARGS="-DCMAKE_MODULE_PATH=$PFX/lib/cmake/OpenVDB -DOpenVDB_ROOT=$PFX -DBLOSC_ROOT=$LD -DTBB_ROOT=$LD -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH" /usr/bin/python3.10 -m pip install --no-build-isolation --no-deps --target /mnt/data/venvs/vdbfusion . 2>&1 | tail -30
