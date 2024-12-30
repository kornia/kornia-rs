#!/usr/bin/env bash

set -ex

apt-get update && apt-get install -y cmake nasm libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev # Install NASM and GStreamer
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin # Install Just for command runner
curl -LsSf https://astral.sh/uv/install.sh | sh # Install UV for manage python virtual environments
rustc --version
