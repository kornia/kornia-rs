#!/usr/bin/env bash

set -ex

apt-get update && apt-get install -y cmake nasm libclang-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev # Install system dependencies
curl -fsSL https://pixi.sh/install.sh | bash # Install Pixi for package and environment management
export PATH="$HOME/.pixi/bin:$PATH"
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin # Install Just for command runner
curl -LsSf https://astral.sh/uv/install.sh | sh # Install UV for manage python virtual environments
rustc --version
