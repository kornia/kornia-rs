#!/usr/bin/env bash

set -ex

apt-get update && apt-get install -y cmake nasm libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev # Install NASM and GStreamer
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin # Install Just for command runner
curl -LsSf https://astral.sh/uv/install.sh | sh # Install UV for manage python virtual environments
rustc --version

# apt-get install -y --no-install-recommends wget gpg lsb-release ca-certificates software-properties-common
# wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
# echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list
# apt-get update
# apt-get install -y intel-oneapi-mkl-devel intel-oneapi-compiler-dpcpp-cpp-runtime
# rm -rf /var/lib/apt/lists/*

